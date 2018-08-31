import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from random import randint


import torch
import numpy as np
from PIL import ImageDraw

from models.dhSegment import dhSegment
from utils.logger import print_normal, print_warning, TerminalColors, print_error
from dataset.icdar_document_set import ICDARDocumentSet, ICDARDocumentEvalSet
from utils.image import save_connected_components
from utils.image import image_numpy_to_pillow, image_numpy_to_pillow_bw
from evaluate import extract, output_baseline, output_image_bloc
from utils.trainer import MovingAverage, CPUParallel


def main():
    parser = argparse.ArgumentParser(description="socr")
    parser.add_argument('--name', type=str, default="dhSegment")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--overlr', action='store_const', const=True, default=False, help="Override the learning rate")
    parser.add_argument('--bs', type=int, default=16, help="The batch size")
    parser.add_argument('--losstype', type=str, default='bce', help="The loss type. Ex : mse, bce, norm")
    parser.add_argument('--thicknesses', type=int, default=2, help="Line thicknesses in the document")
    parser.add_argument('--hystmin', type=float, default=0.5, help="Hysteresys thresholding minimum")
    parser.add_argument('--hystmax', type=float, default=0.5, help="Hysteresys thresholding maximum")
    parser.add_argument('--expdecay', type=float, default=0.98, help="Exponential decay")
    parser.add_argument('--heightimportance', type=float, default=0.001, help="Height prediction importance during the training")
    parser.add_argument('--weightdecay', type=float, default=0.000001, help="Weight decay")
    parser.add_argument('--epochlimit', type=int, default=None, help="Limit the number of epoch")
    parser.add_argument('--bnmomentum', type=float, default=0.1, help="BatchNorm Momentum")
    parser.add_argument('--disablecuda', action='store_const', const=True, default=False, help="Disable cuda")
    parser.add_argument('--icdartrain', type=str, help="Path to the ICDAR Training set")
    parser.add_argument('--icdartest', type=str, default=None, help="Path to the ICDAR Testing set")
    parser.add_argument('--generated', action='store_const', const=True, default=False, help="Enable generated data")
    args = parser.parse_args()

    model = dhSegment(args.losstype, args.hystmin, args.hystmax, args.thicknesses, args.heightimportance, args.bnmomentum)
    loss = model.create_loss()

    if not args.disablecuda:
        model = torch.nn.DataParallel(model.cuda())
        loss = loss.cuda()
    else:
        model = CPUParallel(model.cpu())
        loss = loss.cpu()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    adaptative_optimizer = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.expdecay)

    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_name = "checkpoints/" + args.name + ".pth.tar"

    epoch = 0

    if os.path.exists(checkpoint_name):
        print_normal("Restoring the weights...")
        checkpoint = torch.load(checkpoint_name)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        adaptative_optimizer.load_state_dict(checkpoint['adaptative_optimizer'])
    else:
        print_warning("Can't find '" + checkpoint_name + "'")

    if args.overlr is not None:
        print_normal("Overwriting the lr to " + str(args.lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    train_databases = [ICDARDocumentSet(args.icdartrain, loss, True)]

    if args.generated:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "submodules/scribbler"))
        from scribbler.generator import DocumentGenerator
        train_databases.append(DocumentGenerator(loss))

    train_database = torch.utils.data.ConcatDataset(train_databases)

    test_database_path = None
    if args.icdartest is not None:
        test_database_path = args.icdartest

    moving_average = MovingAverage(max(train_database.__len__() // args.bs, 1024))

    try:
        while True:
            if args.epochlimit is not None and epoch > args.epochlimit:
                print_normal("Epoch " + str(args.epochlimit) + "reached !")
                break

            model.train()

            loader = torch.utils.data.DataLoader(train_database, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=collate)
            for i, data in enumerate(loader, 0):

                inputs, labels = data

                optimizer.zero_grad()

                variable = torch.autograd.Variable(inputs).float()
                labels = torch.autograd.Variable(labels).float()

                if not args.disablecuda:
                    variable = variable.cuda()
                    labels = labels.cuda()
                else:
                    variable = variable.cpu()
                    labels = labels.cpu()

                outputs = model(variable)
                loss_value = loss.forward(outputs, labels)
                loss_value.backward()

                loss_value_cpu = loss_value.data.cpu().numpy()

                optimizer.step()

                loss_value_np = float(loss_value.data.cpu().numpy())
                moving_average.addn(loss_value_np)

                if (i * args.bs) % 8 == 0:
                    sys.stdout.write(TerminalColors.BOLD + '[%d, %5d] ' % (epoch + 1, (i * args.bs) + 1) + TerminalColors.ENDC)
                    sys.stdout.write('lr: %.8f; loss: %.4f ; curr: %.4f ;\r' % (optimizer.state_dict()['param_groups'][0]['lr'], moving_average.moving_average(), loss_value_cpu))

            epoch = epoch + 1
            adaptative_optimizer.step()

            sys.stdout.write("\n")

            try:
                if args.icdartest is not None:
                    callback(model, loss, test_database_path)
            except Exception as e:
                print_error("Can't test : " + str(e))

    except KeyboardInterrupt:
        pass

    print_normal("Done training ! Saving...")
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'adaptative_optimizer': adaptative_optimizer.state_dict(),
    }, checkpoint_name)


def collate(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]

    min_width = min([d.size()[1] for d in data])
    min_height = min([d.size()[0] for d in data])

    min_width = min(min_width, 300)
    min_height = min(min_height, 300)

    new_data = []
    new_label = []

    for i in range(0, len(data)):
        d = data[i]

        crop_x = randint(0, d.size()[1] - min_width)
        crop_y = randint(0, d.size()[0] - min_height)

        d = d[crop_y:crop_y + min_height, crop_x:crop_x + min_width]
        d = torch.transpose(d, 0, 2)
        d = torch.transpose(d, 1, 2)
        new_data.append(d)

        d = label[i]

        d = d[crop_y:crop_y + min_height, crop_x:crop_x + min_width]
        d = torch.transpose(d, 0, 2)
        d = torch.transpose(d, 1, 2)
        new_label.append(d)

    data = torch.stack(new_data)
    label = torch.stack(new_label)

    return [data, label]


def evaluate(model, loss, path):
    """
    Evaluate the line localizator. Output all the results to the 'results' directory.

    :param path: The path of the images, with or without associated XMLs
    """
    print_normal("Evaluating " + path)

    if not os.path.exists("results"):
        os.makedirs("results")

    data_set = ICDARDocumentEvalSet(path, loss)

    loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=1)

    count = 0

    for i, data in enumerate(loader, 0):
        resized, image, path, label = data

        percent = i * 100 // data_set.__len__()
        sys.stdout.write(str(percent) + "%... Processing \r")

        lines, positions, probsmap, components = extract(model, loss, image, resized, with_images=False)

        output_image_bloc(image, positions).save("results/" + str(count) + ".jpg", "JPEG")

        save_connected_components(components, "results/" + str(count) + ".components.jpg")

        image_numpy_to_pillow_bw(probsmap[0].cpu().detach().numpy()).save("results/" + str(count) + ".probs.jpg")
        del probsmap

        image_numpy_to_pillow_bw(label[0][0].cpu().detach().numpy()).save("results/" + str(count) + ".probs.gt.jpg")

        xml_path = os.path.join(os.path.dirname(path[0]), os.path.splitext(os.path.basename(path[0]))[0] + ".xml")
        if not os.path.exists(xml_path):
            xml_path = os.path.join(os.path.dirname(path[0]),
                                    "page/" + os.path.splitext(os.path.basename(path[0]))[0] + ".xml")

        if os.path.exists(xml_path):
            shutil.copy2(xml_path, "results/" + str(count) + ".xml")
            with open("results/" + str(count) + ".txt", "w") as text_file:
                text_file.write(output_baseline(positions))
        else:
            print_warning("Can't find : '" + xml_path + "'")

        count = count + 1


def run_transkribus():
    all_txt = "\n".join([f for f in os.listdir("results") if os.path.isfile(os.path.join("results", f)) and f.endswith(".txt")])
    all_xml = "\n".join([f for f in os.listdir("results") if os.path.isfile(os.path.join("results", f)) and f.endswith(".xml")])

    with open("results/txt.lst", "w") as file:
        file.write(all_txt)

    with open("results/xml.lst", "w") as file:
        file.write(all_xml)

    pipe = subprocess.Popen(["java", "-jar", "../transkribus.jar", "xml.lst", "txt.lst"], stdout=subprocess.PIPE, cwd="results")
    texts = pipe.communicate()
    texts = ["" if text is None else text.decode() for text in texts]
    text = "\n".join(texts)
    return text


def callback(model, loss, test_path):
    model.eval()
    subprocess.run(['rm', '-R', 'results'])

    evaluate(model, loss, test_path)
    result = run_transkribus()
    lines = result.split("\n")
    probs = [line.split(",") for line in lines]
    probs = [[prob.replace(" ", "") for prob in problist] for problist in probs]

    new_probs = []
    total = None

    for i in range(0, len(probs)):
        try:
            id = probs[i][3].split(".")[0]
            if id == "TOTAL":
                total = probs[i]
        except Exception as e:
            pass

    print_normal("P : " + str(total[0]) + "; F : " + str(total[1]) + "; F1 : " + str(total[2]))

    for i in range(0, len(probs)):
        try:
            new_probs.append([float(probs[i][0]), float(probs[i][1]), float(probs[i][2]), probs[i][3], probs[i][4]])
        except:
            pass

    new_probs.sort(key=lambda x: x[2])

    for i in range(0, len(new_probs)):
        id = new_probs[i][3].split(".")[0]
        if id != "TOTAL":
            for ext in [".jpg", ".probs.jpg", ".probs.gt.jpg", ".components.jpg", ".txt", ".xml"]:
                os.rename("results/" + id + ext, 'results/%.4f%s' % (new_probs[i][2], ext))
        else:
            print(new_probs[i])

    return total[2]


if __name__ == '__main__':
    main()