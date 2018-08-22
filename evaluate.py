import os
import shutil
import argparse
import subprocess

import torch
import numpy as np
from PIL import ImageDraw

from utils.dataset import FileDataset
from utils.image import image_numpy_to_pillow, image_numpy_to_pillow_bw, save_connected_components
from models.dhSegment import dhSegment
from utils.trainer import CPUParallel


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



def main():
    parser = argparse.ArgumentParser(description="socr")
    parser.add_argument('paths', metavar='N', type=str, nargs='+')
    parser.add_argument('--name', type=str, default="dhSegment")
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

    subprocess.run(['rm', '-R', 'results'])

    if not os.path.exists("results"):
        os.makedirs("results")

    model = dhSegment(args.losstype, args.hystmin, args.hystmax, args.thicknesses, args.heightimportance, args.bnmomentum)
    loss = model.create_loss()

    if not args.disablecuda:
        model = torch.nn.DataParallel(model.cuda())
        loss = loss.cuda()
    else:
        model = CPUParallel(model.cpu())
        loss = loss.cpu()

    model.eval()

    checkpoint_name = "checkpoints/" + args.name + ".pth.tar"
    assert os.path.exists(checkpoint_name)

    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint['state_dict'])

    data_set = FileDataset()
    for path in args.paths:
        data_set.recursive_list(path)
    data_set.sort()

    print(str(data_set.__len__()) + " files")

    loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)
    count = 0

    for i, data in enumerate(loader, 0):
        resized, image, path = data

        percent = i * 100 // data_set.__len__()
        print(str(percent) + "%... Processing " + path[0])

        positions, lines, probsmap, components = extract(model, loss, image, resized)

        print("Creating output image bloc to" + path[0] + ".bloc.result.jpg")
        output_bloc_image = output_image_bloc(image, positions, lwidth=1)
        output_bloc_image.save("results/" + str(count) + ".bloc.jpg", "JPEG")

        save_connected_components(components, "results/" + str(count) + ".components.jpg")

        image_numpy_to_pillow_bw(probsmap[0].cpu().detach().numpy()).save( "results/" + str(count) + ".probs.jpg")

        os.makedirs("results/" + str(count))

        # for k in range(0, len(lines)):
        #     image_numpy_to_pillow(lines[k]).save("results/" + str(count) + "/" + str(k) + ".jpg")

        xml_path = os.path.join(os.path.dirname(path[0]), os.path.splitext(os.path.basename(path[0]))[0] + ".xml")
        if os.path.exists(xml_path):

            shutil.copy2(xml_path, "results/" + str(count) + ".xml")

            with open("results/" + str(count) + ".txt", "w") as text_file:
                text_file.write(output_baseline(positions))

        else:
            print("Can't find : '" + xml_path + "'")



        count = count + 1

if __name__ == '__main__':
    main()