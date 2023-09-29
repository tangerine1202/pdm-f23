import cv2
import numpy as np

points = []


def xyz2R(x_deg, y_deg, z_deg):
    """
    :param x_rot: rotation angle along x axis
    :param y_rot: rotation angle along y axis
    :param z_rot: rotation angle along z axis
    """
    x_rad, y_rad, z_rad = np.deg2rad(x_deg), np.deg2rad(y_deg), np.deg2rad(z_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x_rad), -np.sin(x_rad)],
        [0, np.sin(x_rad), np.cos(x_rad)]
    ])
    Ry = np.array([
        [np.cos(y_rad), 0, np.sin(y_rad)],
        [0, 1, 0],
        [-np.sin(y_rad), 0, np.cos(y_rad)]
    ])
    Rz = np.array([
        [np.cos(z_rad), -np.sin(z_rad), 0],
        [np.sin(z_rad), np.cos(z_rad), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    return R


class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape
        self.points = points

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image

            # Coordinate system
            # Use right-hand coordinate system
            # W - world coordinate system: (x, y, z) = (left, up, front)
            # C - camera coordinate system: (x, y, z) = (right, down, front)
            # Assumption
            # R, t - known pose of top view and front view
            # K - known camera intrinsic matrix and it is the same for both top view and front view
            # n_bev_C - the normal vector of the plane in bev view
            # d_bev_C - the distance from the origin to the plane in bev view
        """

        x_bev = np.array(self.points).T
        x_bev_h = np.vstack((x_bev, np.full((1, x_bev.shape[1]), 1)))

        t_bev_W = np.array([[0], [2.5], [0]])
        t_front_W = np.array([[0], [1], [0]])
        # Note: watch out the direction of normal vector
        n_bev_C = np.array([[0], [0], [1]])
        d_bev_C = t_bev_W[1]

        # translation from the the front camera to the bev camera
        t_front_bev = t_bev_W - t_front_W

        # rotation from the front camera to the bev camera
        R_bev_front = xyz2R(theta, phi, gamma)
        # rotation from the bev camera to the front camera
        R_front_bev = R_bev_front.T

        # focal_length * tan(fov / 2) = width / 2
        fx = self.width / 2 / np.tan(np.deg2rad(fov / 2))
        fy = self.height / 2 / np.tan(np.deg2rad(fov / 2))
        cx = self.width / 2
        cy = self.height / 2
        K = np.array([
            [-fx, 0, cx],
            [0, -fy, cy],
            [0, 0, 1]
        ])
        K_inv = np.linalg.inv(K)

        # X_bev_C = z_bev * K_inv @ x_bev_h
        # d_bev_C = n_bev_C.T @ X_bev_C[:, 0]

        H = K @ (R_front_bev + t_front_bev * n_bev_C.T / d_bev_C) @ K_inv

        x_front_h = H @ x_bev_h
        x_front_h = x_front_h / x_front_h[2, :]
        x_front = x_front_h[:2, :]

        new_pixels = x_front.T.astype(int)
        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
