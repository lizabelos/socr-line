import argparse
import os
import shutil
import subprocess
import sys

import torch
from PIL import Image, ImageDraw
import numpy as np

from models.dhSegment import dhSegment
from utils.trainer import Trainer
from utils.logger import print_warning, print_normal
from dataset.icdar_document_set import ICDARDocumentSet, ICDARDocumentEvalSet
from utils.image import save_connected_components
from utils.image import image_numpy_to_pillow, image_numpy_to_pillow_bw



def main():
    parser = argparse.ArgumentParser(description="socr")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--overlr', action='store_const', const=True, default=False)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--losstype', type=str, default='bce')
    parser.add_argument('--thicknesses', type=int, default=2)
    parser.add_argument('--hystmin', type=float, default=0.5)
    parser.add_argument('--hystmax', type=float, default=0.5)
    parser.add_argument('--expdecay', type=float, default=0.98)
    parser.add_argument('--heightimportance', type=float, default=0.03)
    parser.add_argument('--weightdecay', type=float, default=0.000001)
    parser.add_argument('--epochlimit', type=int, default=75)
    parser.add_argument('--bnmomentum', type=float, default=0.1)
    parser.add_argument('--disablecuda', action='store_const', const=True, default=False)
    args = parser.parse_args()

    model = dhSegment(args.losstype, args.hystmin, args.hystmax, args.thicknesses, args.heightimportance, args.expdecay, args.bnmomentum)
    loss = model.create_loss()

    if not args.disablecuda:
        model = model.cuda()
        loss = loss.cuda()
    else:
        model = model.cpu()
        loss = loss.cpu()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    trainer = Trainer(model, loss, optimizer, args.name)

    if args.overlr is not None:
        print_normal("Overwriting the lr to " + str(args.lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    train_database = ICDARDocumentSet("/home/tbelos/dataset/icdar/2017-baseline/train-complex", loss, True)
    trainer.train(train_database, batch_size=args.bs, callback=lambda: callback(model, loss, "/home/tbelos/dataset/icdar/2017-baseline/validation-complex"), epoch_limit=args.epochlimit)


def extract(model, loss, original_image, resized_image, with_images=True):
    """
    Extract all the line from the given image

    :param image: A tensor image
    :return: The extracted line, as pillow image, and their positions
    """
    if type(original_image).__module__ == np.__name__:
        original_image = torch.from_numpy(original_image).unsqueeze(0)

    if type(resized_image).__module__ == np.__name__:
        resized_image = torch.from_numpy(resized_image).unsqueeze(0)

    is_cuda = next(model.parameters()).is_cuda

    image = torch.autograd.Variable(resized_image).float()

    if is_cuda:
        image = image.cuda()
    else:
        image = image.cpu()

    image = loss.process_labels(image)
    result = model(torch.autograd.Variable(image))[0]
    lines, components = loss.ytrue_to_lines(original_image.cpu().numpy()[0], result.cpu().detach().numpy(), with_images)

    pillow_lines = [line for line, pos in lines]
    pos = [pos for line, pos in lines]

    return pillow_lines, pos, result, components


def output_image_bloc(image, lines, lwidth=5):
    """
    Draw the lines to the image

    :param image: The image
    :param lines: The lines
    :return: The new image
    """
    image = image_numpy_to_pillow(image.cpu().numpy()[0])
    image = image.convert("L").convert("RGB")
    image_drawer = ImageDraw.Draw(image)
    for i in range(0, len(lines)):
        positions = list(lines[i])
        for i in range(0, len(positions) // 2 - 1):
            image_drawer.line((positions[i * 2], positions[i * 2 + 1], positions[i * 2 + 2], positions[i * 2 + 3]),
                              fill=(128, 0, 0), width=lwidth)

    return image


def output_baseline(lines):
    """
    Output a writable string of the lines

    :param lines: The lines
    :return: The string
    """
    result = ""

    for positions in lines:
        first_time = True
        positions = list(positions)
        for i in range(0, len(positions) // 2):
            if not first_time:
                result = result + ";"
            result = result + str(int(positions[i * 2])) + "," + str(int(positions[i * 2 + 1]))
            first_time = False
        result = result + "\n"

    return result


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