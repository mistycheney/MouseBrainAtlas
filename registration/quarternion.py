import numpy as np

def quarternion_normalization_jacobian(qr,qx,qy,qz):

    qr2 = qr**2
    qx2 = qx**2
    qy2 = qy**2
    qz2 = qz**2
    qxqy = qx*qy
    qxqz = qx*qz
    qyqz = qy*qz
    qrqz = qr*qz
    qrqx = qr*qx
    qrqy = qr*qy

    D = np.array([[qx2+qy2+qz2, -qrqx, -qrqy, -qrqz],
                 [-qrqx, qr2+qy2+qz2, -qxqy, -qxqz],
                 [-qrqy, -qxqy, qr2+qx2+qz2, -qyqz],
                 [-qrqz, -qxqz, -qyqz, qr2+qx2+qy2]])

    J = 1./np.sqrt(qr2+qx2+qy2+qz2)**3*D

    return J


def quarternion_jacobian_points(qr,qx,qy,qz,x,y,z):

    dxdqr = 2*(-qz*y+qy*z)
    dxdqx = 2*qy*y+qz*z
    dxdqy = 2*(-2*qy*x+qx*y+qr*z)
    dxdqz = 2*(-2*qz*x-qr*y+qx*z)

    dydqr = 2*(qz*x-qx*z)
    dydqx = 2*(qy*x-2*qx*y-qr*z)
    dydqy = 2*(qx*x+qz*z)
    dydqz = 2*(qr*x-2*qz*y+qy*z)

    dzdqr = 2*(-qy*x+qx*y)
    dzdqx = 2*(qz*x+qr*y-2*qx*z)
    dzdqy = 2*(-qr*x+qz*y-2*qy*z)
    dzdqz = 2*(qx*x+qy*y)

    J = np.array([[dxdqr, dxdqr, dxdqr, dxdqr]]) * \
           quarternion_normalization_jacobian(qr,qx,qy,qz)

    return J

def quaternion_jacobian(qr,qx,qy,qz,x,y,z):

    J = np.dot(2*np.array([[-qz*y+qy*z, qy*y+qz*z, -2*qy*x+qx*y+qr*z, -2*qz*x-qr*y+qx*z],
                [qz*x-qx*z, qy*x-2*qx*y-qr*z, qx*x+qz*z, qr*x-2*qz*y+qy*z],
                [-qy*x+qx*y, qz*x+qr*y-2*qx*z, -qr*x+qz*y-2*qy*z, qx*x+qy*y]]),
           quarternion_normalization_jacobian(qr,qx,qy,qz))

    return J

def quarternion_to_matrix(qr,qx,qy,qz):
    qr2 = qr**2
    qx2 = qx**2
    qy2 = qy**2
    qz2 = qz**2
    qxqy = qx*qy
    qxqz = qx*qz
    qyqz = qy*qz
    qrqz = qr*qz
    qrqx = qr*qx
    qrqy = qr*qy
    R = np.array([[qr2+qx2-qy2-qz2, 2*(qxqy-qrqz), 2*(qxqz+qrqy)],
                  [2*(qxqy+qrqz), qr2-qx2+qy2-qz2, 2*(qyqz-qrqx)],
                  [2*(qxqz-qrqy), 2*(qyqz+qrqx), qr2-qx2-qy2+qz2]])
    return R
