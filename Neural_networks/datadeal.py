import os
import shutil
import numpy as np
import tarfile
from scipy.io import loadmat

# 用代码前请先下载mini_imagenet

def qiulabel():
    imagepath = "./data/mini_imagenet/images"
    target = "./data/mini_imagenet/images_OK"
    if not os.path.exists(target):
        os.mkdir(target)
    lines = os.listdir(imagepath)
    know = set()
    for line in lines:
        if line.endswith(".DS_Store"):
            continue
        if line[:9] not in know:
            if not os.path.exists(target + "/" + line[:9]):
                os.mkdir(target + "/" + line[:9])
            know.add(line[:9])
        path1 = imagepath + "/" + line
        path2 = target + "/" + line[:9] + "/"
        if not os.path.exists(path2 + line):
            shutil.copy(path1, path2)
    # print(len(lines))


def qiudelimage(rate):
    path1 = "./data/mini_imagenet/images_OK"
    train = "./data/mini_imagenet/del_images/train"
    test = "./data/mini_imagenet/del_images/test"
    trainrate = rate * 0.8
    testrate = rate * 0.2
    if not os.path.exists("./data/mini_imagenet/del_images"):
        os.mkdir("./data/mini_imagenet/del_images")
    if not os.path.exists(train):
        os.mkdir(train)
    else:
        shutil.rmtree(train)
        os.mkdir(train)
    if not os.path.exists(test):
        os.mkdir(test)
    else:
        shutil.rmtree(test)
        os.mkdir(test)
    lines = os.listdir(path1)
    for line in lines:
        path2 = path1 + "/" + line
        lines1 = os.listdir(path2)
        trainnum = int(len(lines1) * trainrate)
        testnum = int(len(lines1) * testrate)
        i = 0
        random_inter = np.random.choice(np.arange(0, len(lines1)), trainnum + testnum, replace=False)
        path3 = train + "/" + line
        os.mkdir(path3)
        while trainnum > 0:
            shutil.copy(path2 + "/" + lines1[random_inter[i]], path3)
            trainnum -= 1
            i += 1
        path3 = test + "/" + line
        os.mkdir(path3)
        while testnum > 0:
            shutil.copy(path2 + "/" + lines1[random_inter[i]], path3)
            testnum -= 1
            i += 1

    # print(len(lines))
def imagenetdeal():#creat train,val
    # tar_file="data/imagenet/ILSVRC2012_img_val.tar"
    # tar_file="data/imagenet/ILSVRC2012_devkit_t12.tar.gz"
    tar_file="data/imagenet/ILSVRC2012_img_train.tar"
    target="data/imagenet/train"
    # target = "data/imagenet/help"
    # target = "data/imagenet/val"
    os.makedirs(target,exist_ok=True)
    with tarfile.open(tar_file,'r') as tar:
        tar.extractall(path=target)
    os.remove(tar_file)
    print("work done")

def imagesort():#sort val
    file=open("data/imagenet/help/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")
    lines=file.readlines()
    target="data/imagenet/sort_val"
    val="data/imagenet/val"
    lines1=os.listdir(val)
    for i in range(1,1001):
        target_file=target+'/'+str(i)
        os.makedirs(target_file,exist_ok=True)
    for i in range(len(lines)):
        target_file=target+"/"+lines[i][:-1]
        shutil.copy(val+"/"+lines1[i],target_file)
    shutil.rmtree(val)
    print("work done")
    # print(len(lines[-1]))
    # print(lines[-1])

def createtrain():
    train="data/imagenet/train"
    lines=os.listdir(train)
    for line in lines:
        name=line.split(".")[0]
        tar_file=train+"/"+line
        target=train+"/"+name
        os.makedirs(target,exist_ok=True)
        with tarfile.open(tar_file,'r') as tar:
            tar.extractall(path=target)
        os.remove(tar_file)
    print("work done")
    # print(len(lines),lines[0].split(".")[0])

def valnamechange():
    mat=loadmat("data/imagenet/help/ILSVRC2012_devkit_t12/data/meta.mat")
    val="data/imagenet/sort_val"
    lines=os.listdir(val)
    for line in lines:
        idx=int(line)
        oldname=val+"/"+line
        newname=val+"/"+str(mat['synsets'][idx-1][0][1][0])
        os.rename(oldname,newname)
    print("work done")
    # print(mat['synsets'][0][0][1][0])

if __name__ == "__main__":
    # qiulabel()
    # qiudelimage(1)
    # imagenetdeal()
    # imagesort()
    # createtrain()
    # valnamechange()
    print()