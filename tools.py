# Utility functions

# Imports

from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def line_plot(args, data, title):
    plt.rcParams["figure.figsize"] = (10, 8)
    sns.lineplot(data=data)
    plt.grid()
    plt.title(title)
    if args.experiment == "classic":
        plt.xticks(range(args.eval_split))
    else:
        plt.xticks(range(int(args.eval_split / 5)))
    plt.xlabel("Test domain")
    plt.ylabel("Accuracy")
    plt.savefig(f"output/{args.name}/{title}.jpg")
    plt.show()


def rotate_digits(dataset, n_domains: int, step_size: int):
    # Parameters
    # 1. Dataset with samples of type (Image, int)
    # 2. Number of domains
    # 3. Step size in degrees

    # Returns
    # 1. dictionary, rotated_images that contains rotation angles as keys and lists of images as values
    #    pertaining to each key
    # 2. List of labels of said images

    # Calculate the angles for required number of domains for given step size
    angles = [x * step_size for x in range(n_domains)]
    # Creating a dictionary of lists to store images indexed by their rotation angles
    rotated_images = {angle: [] for angle in angles}
    # List of labels of images ordered according to the same list indices
    labels = []

    # Transform to tensor function
    tensorify = transforms.Compose([transforms.ToTensor()])

    dataset_size = len(dataset)
    for i in tqdm(range(dataset_size)):
        img, l = dataset[i]
        labels.append(l)
        r_images = [img.rotate(angle) for angle in angles]
        for i in range(n_domains):
            rotated_images[angles[i]].append(tensorify(r_images[i]))

    return rotated_images, labels


def visualise_across_domains(images: dict, labels: list, idx: int):
    angles = list(images.keys())
    n_domains = len(angles)
    figure = plt.figure(figsize=(10, 10 * n_domains))

    for i in range(n_domains):
        figure.add_subplot(1, n_domains, i + 1)
        plt.title(f"{angles[i]} deg")
        plt.axis("off")
        plt.imshow(images[angles[i]][idx], cmap="gray")

    plt.show()
    print(f"Label : {labels[idx]}")


def split_tasks(images: list, labels: list, classes_per_task: int = 2):
    # Parameters
    # 1. List of images pertaining to a singular domain
    # 2. List of labels
    # 3. Number of classes per task

    n_images = len(images)
    n_classes = len(set(labels))
    n_splits = n_classes // classes_per_task
    task_images = {task: [] for task in range(n_splits)}
    task_labels = {task: [] for task in range(n_splits)}
    c = list(range(classes_per_task))
    for i in range(n_splits):
        # Labels to be included
        l = [x + (i * classes_per_task) for x in c]
        for j in range(n_images):
            if labels[j] in l:
                task_labels[i].append(labels[j])
                task_images[i].append(images[j])
    return task_images, task_labels
