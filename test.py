import glob
import numpy as np
import cv2
import process
import utils


def make_sun(color_img: np.ndarray, bin_img: np.ndarray) -> process.Sun:
    sun: process.Sun = process.Sun(color_img, bin_img)
    sunspots: list = utils.detect_sunspots(bin_img)
    sun.SetSunspots(sunspots)
    return sun


def main():
    # 画像の読み込み
    img_paths: list[str] = glob.glob("img/*.jpg")
    imgs: list[np.ndarray] = utils.read_imgs(img_paths)

    # 二値化
    imgs_bin: list[np.ndarray] = utils.imgs_binarization(imgs)

    # 背景の黒い部分をトリミングして正方形にする
    (color_imgs, bin_imgs) = utils.clip_imgs(imgs, imgs_bin)

    # Sunクラスにまとめる
    suns: list[process.Sun] = []
    for cimg, bimg in zip(color_imgs, bin_imgs):
        try:
            sun = make_sun(cimg, bimg)
        except:
            pass
        suns.append(sun)

    cv2.imshow("img", suns[0].GetImage("color"))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
