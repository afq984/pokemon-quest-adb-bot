import subprocess
import io
import os
import time

import numpy
import cv2


def capture():
    out = subprocess.check_output(['adb', 'exec-out', 'screencap', '-p'])
    a = numpy.frombuffer(out, dtype=numpy.uint8)
    return cv2.imdecode(a, cv2.IMREAD_COLOR)


def tap(x, y):
    subprocess.check_call(['adb', 'shell', 'input', 'tap', str(x), str(y)])


class CapTool:
    def __init__(self, im, window='Capture Tool'):
        self.m = 2
        self.im = im
        self.window = window
        self.pos0 = None
        self.pos1 = None
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self.mouseCallback)
        self.resized = cv2.resize(self.im, None, fx=1/self.m, fy=1/self.m)
        cv2.imshow(window, self.resized)

    def mouseCallback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.pos0 = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.pos1 = (x, y)
        else:
            return
        if self.pos0 is not None and self.pos1 is not None:
            demo = self.resized.copy()
            cv2.rectangle(demo, self.pos0, self.pos1, (255, 255, 255), 2)
            cv2.imshow(self.window, demo)
            print(f'({self.pos1[0] - self.pos0[0]:+d}, {self.pos1[1] - self.pos0[1]:+d})')

    def x_or_y(self, x0y1):
        if self.pos0 is None or self.pos1 is None:
            a = 0
            b = self.im.shape[1 - x0y1]
        else:
            a = self.pos0[x0y1]
            b = self.pos1[x0y1]
        return slice(self.m * min(a, b), self.m * max(a, b))

    @property
    def x(self):
        return self.x_or_y(0)

    @property
    def y(self):
        return self.x_or_y(1)


def captool():
    im = capture()
    ct = CapTool(im)
    while True:
        key = cv2.waitKey(1)
        if key == ord('\n'):
            break
        elif key == 27:
            return
    cv2.imwrite('crop.png', im[ct.y, ct.x, ...])


def match(im, template):
    res = cv2.matchTemplate(im, template, cv2.TM_SQDIFF_NORMED)
    val, _, pos0, _ = cv2.minMaxLoc(res)
    # _, val, _, pos0 = cv2.minMaxLoc(res)
    h, w = template.shape[:-1]
    pos1 = (pos0[0] + w, pos0[1] + h)
    return val, pos0, pos1


def play():
    templates = []
    preconditions = []
    for i in range(8):
        template = cv2.imread(f'{i}.png')#, cv2.IMREAD_GRAYSCALE)
        precond_image = f'{i}p.png'
        if os.path.exists(precond_image):
            templatep = cv2.imread(precond_image)#, cv2.IMREAD_GRAYSCALE)
        else:
            templatep = template
        templates.append(template)
        preconditions.append(templatep)
    while True:
        im_prev = capture()
        while True:
            time.sleep(1)
            im = capture()
            diff =  (im_prev - im).mean()
            if diff < 10:
                break
            im_prev = im
        for iid, (template, templatep) in enumerate(zip(templates, preconditions)):
            val, a0, a1 = match(im, templatep)
            if val < 0.01:
                val, b0, b1 = match(im, template)
                assert val < 0.01, iid
                x0, y0 = b0
                x1, y1 = b1
                x = (x0 + x1) // 2
                y = (y0 + y1) // 2
                if iid == 0:
                    y += 170
                print(f'match={iid}, tap={x},{y}')
                tap(x, y)
                # cv2.rectangle(im, pos0, pos1, (255, 255, 255), 2)
                # cv2.imshow('im', im)
                if iid in {6, 7}:
                    return
                break
        time.sleep(2)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    options = {
        'captool': captool,
        'play': play,
    }
    parser.add_argument(
        'action', default='play', choices=list(options), nargs='?')
    args = parser.parse_args()
    options[args.action]()


if __name__ == '__main__':
    main()
