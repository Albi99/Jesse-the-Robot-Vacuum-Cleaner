import sys


if __name__ == "__main__":

    arg = sys.argv[1]

    if arg == 'test':
        from rlrc.test import test_shit, test_canvas, test_all_maps
        # test_shit()
        # test_canvas()
        test_all_maps()
    
    elif arg == 'joystick':
        from rlrc.joystick import joystick
        joystick()

    elif arg == 'nn':
        from rlrc.train import train
        train()

    else:
        print('INVALID ARG !')
        print('Choose between: test, joystick, nn.')