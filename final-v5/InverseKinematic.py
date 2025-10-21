import serial
import time
from sympy import *

# need to mannually calculate
d1 = 70
a2 = 125
a3 = 125
a4 = 90
d5 = 150

s = serial.Serial('/dev/tty.usbmodem21201', 115200, timeout=5)


def radian_to_degree(radian):
    return radian / pi * 180


def BraccioInverse(w):
    # w is a vector: (x, y, z, Rot.around x0, Rot.around y0, Rot.around z0)

    # define q as the returned vector of the function: (q1, q2, q3, q4, q5)
    q = Matrix([0, 0, 0, 0, 0, 0])

    # atan2(y,x) function is the 4-quadrant variant for the arcus-tangent function
    q[1] = atan2(w[2], w[1])

    # q234 = q[2] + q[3] + q[4] = w[4]
    q234 = atan2(-w[4] * cos(q[1]) - w[5] * sin(q[1]), -w[6])

    # 2 intermediate variables are introduced: b1 and b2
    b1 = w[1] * cos(q[1]) + w[2] * sin(q[1]) - a4 * cos(q234) + d5 * sin(q234)
    b2 = d1 - a4 * sin(q234) - d5 * cos(q234) - w[3]
    bb = (b1 ** 2) + (b2 ** 2)

    q[3] = acos((bb - a2 ** 2 - a3 ** 2) / (2 * a2 * a3))
    q[2] = atan2((a2 + a3 * cos(q[3])) * b2 - a3 * b1 * sin(q[3]),
                 (a2 + a3 * cos(q[3])) * b1 + a3 * b2 * sin(q[3]))
    q[4] = q234 - q[2] - q[3]

    q[5] = q[1] + pi * log(sqrt(w[4] ** 2 + w[5] ** 2 + w[6] ** 2))
    q[5] = q[5] % pi

    # Offset
    # q[1] = q[1] + 0
    q[2] = q[2] + pi
    q[3] = q[3] + pi / 2
    q[4] = q[4] + pi / 2
    # q[5] = q[5] + 0

    return q


def Braccio_toString(q):
    q_degree = N(radian_to_degree(q))

    for angle in q_degree:
        # use "try" to keep the code running
        try:
            if angle < 0 or angle > 180:
                return None
        except:
            return None

    command = "P" + str(int(q_degree[1])) + ", " + str(int(q_degree[2])) + ", " + str(int(q_degree[3])) + ", " + str(
        int(q_degree[4])) + ", " + str(int(q_degree[5]))

    return command


def grasp_execute(x_coord, y_coord, angle):
    w = Matrix([0, x_coord, y_coord, 0, 0, 0, -exp(angle / 180)])
    command_grasp = Braccio_toString(BraccioInverse(w))
    w2 = Matrix([0, x_coord, y_coord, 20, 0, 0, -exp(angle / 180)])
    command_above = Braccio_toString(BraccioInverse(w2))

    if command_grasp and command_above:
        s.write(b'P90,78,83,90,90,0,50\n')

        # move above
        command_above_1 = command_above + ", 0, 10\n"
        print(command_above_1)
        s.write(command_above_1.encode('ascii'))
        print(s.readline().decode())

        # move downwards
        command_grasp_1 = command_grasp + ", 0, 10\n"
        print(command_grasp_1)
        s.write(command_grasp_1.encode('ascii'))
        print(s.readline().decode())

        # grasp
        command_grasp_2 = command_grasp + ", 80, 10\n"
        print(command_grasp_2)
        s.write(command_grasp_2.encode('ascii'))
        print(s.readline().decode())
    else:
        print("The destination is not in the range")


def move_to_destination_execute():
    s.write(b'P0,78,173,90,0,80,50\n')
    print(s.readline().decode())
    s.write(b'P0,78,173,90,0,0,50\n')
    print(s.readline().decode())


def inverse_kinematic_grasp(x_coord, y_coord, angle):
    s.write(b'P0,78,173,90,0,80,50\n')
    print(s.readline().decode())
    grasp_execute(x_coord, y_coord, angle)
    move_to_destination_execute()
