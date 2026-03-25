from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Tuple, Union
import zipfile
import rarfile
import py7zr
import torchvision
from PIL import Image, ImageOps
from io import BytesIO
from subprocess import getoutput as shell_stdout
import os
import errno
import torch
import sys


def warn(*args: List[Any]) -> None:
    sys.stderr.write(" ".join([str(arg) for arg in args]) + "\n")


def _get_dict(
    compressed_stream: Union[py7zr.SevenZipFile, rarfile.RarFile, zipfile.ZipFile], filter_gt: bool = False, filter_nongt: bool = False
) -> Dict[str, IO]:
    assert not (filter_gt and filter_nongt)

    def isimage(x: str) -> bool:
        return x.split(".")[-1].lower() in ["tiff", "bmp", "jpg", "tif", "jpeg", "png"] and not "skel" in x.lower()

    def isgt(x: str) -> bool:
        return "gt" in x or "GT" in x

    if isinstance(compressed_stream, py7zr.SevenZipFile):
        compressed_stream.reset()
        res_dict = compressed_stream.readall()
        names = res_dict.keys()
        if filter_gt:
            names = [n for n in names if not isgt(n)]
        if filter_nongt:
            names = [n for n in names if isgt(n)]
        return {name: res_dict[name] for name in names}
    elif isinstance(compressed_stream, rarfile.RarFile) or isinstance(compressed_stream, zipfile.ZipFile):
        names = compressed_stream.namelist()
        names = [n for n in names if isimage(n)]
        if filter_gt:
            names = [n for n in names if not isgt(n)]
        if filter_nongt:
            names = [n for n in names if isgt(n)]
        return {name: BytesIO(compressed_stream.read(compressed_stream.getinfo(name))) for name in names}
    else:
        raise ValueError("Filedescriptor must be one of [rar, zip, 7z]")


def extract(archive: str, root: Union[str, None] = None) -> None:
    if archive.endswith(".tar.gz"):
        if root is None:
            cmd = "tar -xpvzf {}".format(archive)
        else:
            cmd = f"mkdir -p {root};tar -xpvzf {archive} -C{root}"
        _ = shell_stdout(cmd)
    else:
        raise NotImplementedError()


def check_os_dependencies() -> bool:
    program_list = ["wget"]
    return all([shell_stdout("which " + prog) for prog in program_list])


def mkdir_p(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resumable_download(url: str, save_dir: str) -> str:
    mkdir_p(save_dir)
    download_cmd = "wget --directory-prefix=%s -c %s" % (save_dir, url)
    warn("Downloading {} ... ".format(url))
    shell_stdout(download_cmd)
    warn("done")
    return os.path.join(save_dir, url.split("/")[-1])


dibco_transform_gray_input = torchvision.transforms.Compose(
    [
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        # lambda x: torch.cat([x, 1 - x])
    ]
)

dibco_transform_color_input = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)

dibco_transform_gt = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor(), lambda x: torch.cat([x, 1 - x])])


class Dibco:
    """Provides one or more of the `DIBCO <https://vc.ee.duth.gr/dibco2019/>` datasets.

    Os dependencies: Other than python packages, unrar and arepack CLI tools must be installed.
    In Ubuntu they can be installed with: sudo apt install unrar atool p7zip-full
    In order to concatenate two DIBCO datasets just add them:
    .. source :: python -> 'Dibco'

        trainset = dibco.Dibco.Dibco2009() + dibco.Dibco.Dibco2013()
        valset = dibco.Dibco.Dibco2017() + dibco.Dibco.Dibco209()

    Each item is a tuple of an RGB PIL image and an Binary PIL image. The images are transformed by ``input_transform``
    and ``gt_transform``.
    """

    urls = {
        "2009_HW": [
            "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBC02009_Test_images-handwritten.rar",
            "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009-GT-Test-images_handwritten.rar",
        ],
        "2009_P": [
            "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009_Test_images-printed.rar",
            "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009-GT-Test-images_printed.rar",
        ],
        "2010": [
            "http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/H_DIBCO2010_test_images.rar",
            "http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/H_DIBCO2010_GT.rar",
        ],
        "2011_P": ["http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/DIBCO11-machine_printed.rar"],
        "2011_HW": ["http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/DIBCO11-handwritten.rar"],
        "2012": ["http://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/dataset/H-DIBCO2012-dataset.rar"],
        "2013": ["http://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/dataset/DIBCO2013-dataset.rar"],
        "2014": [
            "http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/original_images.rar",
            "http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/GT.rar",
        ],
        "2016": [
            "https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-original.zip",
            "https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-GT.zip",
        ],
        "2017": ["https://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_Dataset.7z", "https://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_GT.7z"],
        "2018": ["http://vc.ee.duth.gr/h-dibco2018/benchmark/dibco2018_Dataset.zip", "http://vc.ee.duth.gr/h-dibco2018/benchmark/dibco2018-GT.zip"],
        "2019A": [
            "https://vc.ee.duth.gr/dibco2019/benchmark/dibco2019_dataset_trackA.zip",
            "https://vc.ee.duth.gr/dibco2019/benchmark/dibco2019_gt_trackA.zip",
        ],
        "2019B": [
            "https://vc.ee.duth.gr/dibco2019/benchmark/dibco2019_dataset_trackB.zip",
            "https://vc.ee.duth.gr/dibco2019/benchmark/dibco2019_GT_trackB.zip",
        ],
    }

    urls = {
        "2009_HW": [
            "http://rr.visioner.ca/assets/dibco_mirror/DIBC02009_Test_images-handwritten.rar",
            "http://rr.visioner.ca/assets/dibco_mirror/DIBCO2009-GT-Test-images_handwritten.rar",
        ],
        "2009_P": [
            "http://rr.visioner.ca/assets/dibco_mirror/DIBCO2009_Test_images-printed.rar",
            "http://rr.visioner.ca/assets/dibco_mirror/DIBCO2009-GT-Test-images_printed.rar",
        ],
        "2010": ["http://rr.visioner.ca/assets/dibco_mirror/H_DIBCO2010_test_images.rar", "http://rr.visioner.ca/assets/dibco_mirror/H_DIBCO2010_GT.rar"],
        "2011_P": ["http://rr.visioner.ca/assets/dibco_mirror/DIBCO11-machine_printed.rar"],
        "2011_HW": ["http://rr.visioner.ca/assets/dibco_mirror/DIBCO11-handwritten.rar"],
        "2012": ["http://rr.visioner.ca/assets/dibco_mirror/H-DIBCO2012-dataset.rar"],
        "2013": ["http://rr.visioner.ca/assets/dibco_mirror/DIBCO2013-dataset.rar"],
        "2014": ["http://rr.visioner.ca/assets/dibco_mirror/original_images.rar", "http://rr.visioner.ca/assets/dibco_mirror/GT.rar"],
        "2016": [
            "http://rr.visioner.ca/assets/dibco_mirror/DIBCO2016_dataset-original.zip",
            "http://rr.visioner.ca/assets/dibco_mirror/DIBCO2016_dataset-GT.zip",
        ],
        "2017": ["http://rr.visioner.ca/assets/dibco_mirror/DIBCO2017_Dataset.7z", "http://rr.visioner.ca/assets/dibco_mirror/DIBCO2017_GT.7z"],
        "2018": ["http://rr.visioner.ca/assets/dibco_mirror/dibco2018_Dataset.zip", "http://rr.visioner.ca/assets/dibco_mirror/dibco2018-GT.zip"],
        "2019A": [
            "http://rr.visioner.ca/assets/dibco_mirror/dibco2019_dataset_trackA.zip",
            "http://rr.visioner.ca/assets/dibco_mirror/dibco2019_gt_trackA.zip",
        ],
        "2019B": [
            "http://rr.visioner.ca/assets/dibco_mirror/dibco2019_dataset_trackB.zip",
            "http://rr.visioner.ca/assets/dibco_mirror/dibco2019_GT_trackB.zip",
        ],
    }

    @staticmethod
    def load_single_stream(compressed_stream: Union[py7zr.SevenZipFile, rarfile.RarFile, zipfile.ZipFile]) -> Dict[str, Tuple[IO, IO]]:
        input_name2bs = _get_dict(compressed_stream, filter_gt=True)
        gt_name2bs = _get_dict(compressed_stream, filter_nongt=True)
        id2gt = {n.split("/")[-1].split("_")[0].split(".")[0]: Image.open(fd).copy() for n, fd in gt_name2bs.items()}
        id2in = {n.split("/")[-1].split("_")[0].split(".")[0]: Image.open(fd).copy() for n, fd in input_name2bs.items()}
        assert set(id2gt.keys()) == set(id2in.keys())
        # id2gt = {k: ImageOps.invert(v.convert("RGB")).convert('1') for k, v in id2gt.items()}
        id2in = {k: v.convert("RGB") for k, v in id2in.items()}
        return {k: (id2in[k], id2gt[k]) for k in id2gt.keys()}

    @staticmethod
    def load_double_stream(input_compressed_stream: IO, gt_compressed_stream: IO) -> Dict[str, Tuple[IO, IO]]:
        input_name2bs = _get_dict(input_compressed_stream)
        gt_name2bs = _get_dict(gt_compressed_stream)
        id2in = {n.split("/")[-1].split("_")[0].split(".")[0]: Image.open(fd).copy() for n, fd in input_name2bs.items()}
        id2gt = {n.split("/")[-1].split("_")[0].split(".")[0]: Image.open(fd).copy() for n, fd in gt_name2bs.items()}
        assert set(id2gt.keys()) == set(id2in.keys())
        # id2gt = {k: ImageOps.invert(v.convert("RGB")).convert('1') for k, v in id2gt.items()}
        id2in = {k: v.convert("RGB") for k, v in id2in.items()}
        return {k: (id2in[k], id2gt[k]) for k in id2gt.keys()}

    @staticmethod
    def Dibco2009(**kwargs) -> "Dibco":
        kwargs["partitions"] = ["2009_HW", "2009_P"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2010(**kwargs) -> "Dibco":
        kwargs["partitions"] = ["2010"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2011(**kwargs) -> "Dibco":
        kwargs["partitions"] = ["2011_P", "2011_HW"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2012(**kwargs) -> "Dibco":
        kwargs["partitions"] = ["2012"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2013(**kwargs) -> "Dibco":
        kwargs["partitions"] = ["2013"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2014(**kwargs) -> "Dibco":
        kwargs["partitions"] = ["2014"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2016(**kwargs) -> "Dibco":
        kwargs["partitions"] = ["2016"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2017(**kwargs) -> "Dibco":
        kwargs["partitions"] = ["2017"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2018(**kwargs) -> "Dibco":
        kwargs["partitions"] = ["2018"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2019(**kwargs) -> "Dibco":
        kwargs["partitions"] = ["2019A", "2019B"]
        return Dibco(**kwargs)

    def __init__(
        self,
        partitions: List[str] = ["2009_HW", "2009_P"],
        root: bool = "./tmp/dibco",
        input_transform: Callable[[torch.Tensor], torch.Tensor] = dibco_transform_gray_input,
        gt_transform: Callable[[torch.Tensor], torch.Tensor] = dibco_transform_gt,
        add_mask: bool = False,
    ):
        self.input_transform = input_transform
        self.gt_transform = gt_transform
        self.root = root
        self.add_mask = add_mask
        data = {}
        if "2009" in partitions:
            partitions.remove("2009")
            partitions += ["2009_HW", "2009_P"]
        if "2011" in partitions:
            partitions.remove("2011")
            partitions += ["2011_HW", "2011_P"]
        if "2019" in partitions:
            partitions.remove("2019")
            partitions += ["2019A", "2019B"]
        if "all" in partitions:
            partitions = list(Dibco.urls.keys())
        for partition in partitions:
            for url in Dibco.urls[partition]:
                archive_fname = root + "/" + url.split("/")[-1]
                if not os.path.isfile(archive_fname):
                    resumable_download(url, root)
                else:
                    warn(archive_fname, " found in cache.")
            if len(Dibco.urls[partition]) == 2:
                if Dibco.urls[partition][0].endswith(".rar"):
                    input_rar = rarfile.RarFile(root + "/" + Dibco.urls[partition][0].split("/")[-1])
                    gt_rar = rarfile.RarFile(root + "/" + Dibco.urls[partition][1].split("/")[-1])
                    samples = {partition + "/" + k: v for k, v in Dibco.load_double_stream(input_rar, gt_rar).items()}
                    data.update(samples)
                elif Dibco.urls[partition][0].endswith(".zip") or Dibco.urls[partition][0].endswith(".7z"):
                    zip_input_fname = root + "/" + Dibco.urls[partition][0].split("/")[-1]
                    zip_gt_fname = root + "/" + Dibco.urls[partition][1].split("/")[-1]
                    if zip_input_fname.endswith("7z"):
                        input_zip = py7zr.SevenZipFile(zip_input_fname)
                    else:
                        input_zip = zipfile.ZipFile(zip_input_fname)
                    if zip_gt_fname.endswith("7z"):
                        gt_zip = py7zr.SevenZipFile(zip_gt_fname)
                    else:
                        gt_zip = zipfile.ZipFile(zip_gt_fname)
                    samples = {partition + "/" + k: v for k, v in Dibco.load_double_stream(input_zip, gt_zip).items()}
                    data.update(samples)
                else:
                    raise ValueError("Unknown file type")
            else:
                if Dibco.urls[partition][0].endswith(".rar"):
                    input_rar = rarfile.RarFile(root + "/" + Dibco.urls[partition][0].split("/")[-1])
                    samples = {partition + "/" + k: v for k, v in Dibco.load_single_stream(input_rar).items()}
                    data.update(samples)
                elif Dibco.urls[partition][0].endswith(".zip") or Dibco.urls[partition][0].endswith(".7z"):
                    zip_input_fname = root + "/" + Dibco.urls[partition][0].split("/")[-1]
                    if zip_input_fname.endswith("7z"):
                        # zip_input_fname = zip_input_fname[:-2] + "zip"
                        input_zip = py7zr.SevenZipFile(zip_input_fname)
                    else:
                        input_zip = zipfile.ZipFile(zip_input_fname)
                    samples = {partition + "/" + k: v for k, v in Dibco.load_single_stream(input_zip).items()}
                    data.update(samples)
                else:
                    raise ValueError("Unknown file type")
        id_data = list(data.items())
        self.sample_ids = [sample[0] for sample in id_data]
        self.inputs = [sample[1][0] for sample in id_data]
        self.gt = [sample[1][1] for sample in id_data]

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_img = self.input_transform(self.inputs[item])
        gt = self.gt_transform(self.gt[item])
        if self.add_mask:
            return input_img, gt, torch.ones_like(input_img[:1, :, :])
        else:
            return input_img, gt

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __add__(self, other: "Dibco") -> "Dibco":
        res = Dibco(partitions=[])
        res.root = self.root
        res.input_transform = self.input_transform
        res.gt_transform = self.gt_transform
        res.sample_ids = self.sample_ids + other.sample_ids
        res.inputs = self.inputs + other.inputs
        res.gt = self.gt + other.gt
        return res


all_dibco_keys = set(Dibco.urls.keys())

l1out_partitions = {}
for k in all_dibco_keys:
    trainset = sorted(all_dibco_keys - set([k]))
    trainset, validationset = trainset[:-1], trainset[-1:]
    l1out_partitions[k] = {"train": trainset, "val": validationset, "test": k}


def main_test_dibco():
    from .dibco import Dibco
    from .segvis import segmentation_outputs_to_rgb, save_png_with_metadata
    from torch_labeled_pooling.bunet import MultiheadBUNet
    import torch
    import numpy as np
    import fargv
    import tqdm
    from pathlib import Path
    import sys

    p = {
        "model_path": "./experiments/models/simple_metric0_medium-fg-fonts-sizes.pt",
        "device": "cuda",
        "dibco_name": "all",
        "dump_dir": "/tmp/dibco_results/",
        "save_output": True,
        "save_confusion": False,
        "save_input": False,
        "save_target": False,
        "report_per_sample": False,
        "output_log_path": "stdout",
    }
    args, _ = fargv.fargv(p)
    if args.output_log_path not in ["stdout", "stderr"]:
        logfile = open(args.output_log_path, "a")
    elif args.output_log_path == "stdout":
        logfile = sys.stdout
    elif args.output_log_path == "stderr":
        logfile = sys.stderr
    else:
        raise ValueError("Invalid output log path")
    dibco_partitions = args.dibco_name.split(",")
    model = MultiheadBUNet.resume(args.model_path)[0]
    model.to(device=args.device)
    if args.dump_dir != "":
        Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
    if args.report_per_sample:
        header = f"{'DIBCO':<10} & {'Sample':<10} & {'Accuracy':<10} & {'Precision':<10} & {'Recall':<10} & {'Fscore':<10} \\\\"
        print(header, file=logfile)
    else:
        header = f"{'DIBCO':<10} & {'Accuracy':<1main_inference0} & {'Precision':<10} & {'Recall':<10} & {'Fscore':<10} \\\\"
        print(header, file=logfile)

    total_precisions = []
    total_recalls = []
    total_fscores = []
    total_accuracies = []

    if args.report_per_sample:
        sample_id_str = " &         NA "
    else:
        sample_id_str = ""

    with torch.no_grad():
        for partition in tqdm.tqdm(dibco_partitions):
            ds = Dibco(partitions=dibco_partitions, add_mask=False)
            dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
            partition_precisions = []
            partition_recalls = []
            partition_fscores = []
            partition_accuracies = []
            for n, (sample_input, sample_target) in enumerate(dataloader):
                input_img, target = sample_input.to(args.device), sample_target.to(args.device)
                print(f"{input_img.size()} {target.size()} {input_img.min()} {input_img.max()} {target.min()} {target.max()}")
                output = model(input_img, head_name="fg") > 0.5
                output, target = output[:, :1, :, :].to(torch.uint8), target[:, 1:, :, :].to(torch.uint8)
                confusion_image = output + 2 * target
                vals, confusion_freq = torch.unique(confusion_image, return_counts=True)
                all_freq = torch.zeros(4, dtype=torch.int64, device=args.device)
                all_freq[vals.long()] = confusion_freq
                tp, tn, fp, fn = all_freq[3].item(), all_freq[0].item(), all_freq[1].item(), all_freq[2].item()
                pr = tp / (tp + fp)
                rec = tp / (tp + fn)
                fs = 2 * pr * rec / (pr + rec)
                acc = (tp + tn) / (tp + tn + fp + fn)
                partition_precisions.append(pr)
                partition_recalls.append(rec)
                partition_fscores.append(fs)
                partition_accuracies.append(acc)
                if args.report_per_sample:
                    print(f"{partition:<10} & {n:10d} & {100 * acc:.2f} & {100 * pr:10.2f} & {100 * rec:10.2f}, fscore: {100 * fs:.2f} \\\\", file=logfile)
                sample_name = f"{partition}_{n:04d}"
                if args.dump_dir != "":
                    if args.save_output:
                        save_png_with_metadata(255 - (output[0, 0, :, :].cpu().numpy() * 255), f"{args.dump_dir}/{sample_name}.bin.png")
                    if args.save_confusion:
                        visble_confusion = segmentation_outputs_to_rgb(confusion_image).cpu().numpy()[0, :, :, :].swapaxes(0, 2).swapaxes(0, 1)
                        save_png_with_metadata(visble_confusion, f"{args.dump_dir}/{sample_name}_confusion.png")
                    if args.save_input:
                        save_png_with_metadata(255 - ((255 * input_img[0, 0, :, :]).to(torch.uint8).cpu().numpy()), f"{args.dump_dir}/{sample_name}_input.png")
                    if args.save_target:
                        save_png_with_metadata(255 - (target[0, 0, :, :].cpu().numpy() * 255), f"{args.dump_dir}/{sample_name}_target.png")
                total_accuracies += partition_accuracies
                total_precisions += partition_precisions
                total_recalls += partition_recalls
                total_fscores += partition_fscores
                acc = np.mean(partition_accuracies)
                pr = np.mean(partition_precisions)
                rec = np.mean(partition_recalls)
                fs = np.mean(partition_fscores)
                print(f"{partition:<10} {sample_id_str} & {100 * acc:.2f} & {100 * pr:10.2f} & {100 * rec:10.2f}, fscore: {100 * fs:.2f} \\\\", file=logfile)

            acc = np.mean(total_accuracies)
            pr = np.mean(total_precisions)
            rec = np.mean(total_recalls)
            fs = np.mean(total_fscores)

            if args.report_per_sample:
                sample_id_str = f" & {'NA':<10} "
            else:
                sample_id_str = ""
            print(f"{'Average':<10} {sample_id_str} & {100 * acc:.2f} & {100 * pr:10.2f} & {100 * rec:10.2f}, fscore: {100 * fs:.2f} \\\\", file=logfile)
