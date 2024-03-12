"""
DSC 20 Mid-Quarter Project
Name: Saket Arora
PID:  A16687717
"""

# Part 1: RGB Image #
class RGBImage:
    """
    Creates the RGBImage object class.
    """

    def __init__(self, pixels):
        """
        Initializes RGBImage and Instance Variables
        """
        self.pixels = pixels

    def size(self):
        """
        Returns size of the image
        """
        return (len(self.pixels[0]), len(self.pixels[0][0]))

    def get_pixels(self):
        """
        Returns deep copy of the image
        """
        deep_copy = []
        for i in self.pixels:
            colorlist = []
            for j in i:
                colorlist.append(list(j))
            deep_copy.append(colorlist)
        return deep_copy

    def copy(self):
        """
        Returns deep copy of RGBImage Instance
        """
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        Returns the color of the pixel at a given row and column
        """
        size = self.size()
        assert row in range(size[0])
        assert col in range(size[1])
        index = 2
        return (self.pixels[0][row][col], self.pixels[1][row][col], \
        self.pixels[index][row][col])


    def set_pixel(self, row, col, new_color):
        """
        Uodates color of pixel at a given row and column
        """
        size = self.size()
        assert row in range(size[0])
        assert col in range(size[1])
        index = 2
        if new_color[0] != -1:
            self.pixels[0][row][col] = new_color[0]
        if new_color[1] != -1:
            self.pixels[1][row][col] = new_color[1]
        if new_color[index] != -1:
            self.pixels[index][row][col] = new_color[index]




# Part 2: Image Processing Methods #
class ImageProcessing:
    """
    Contains methods that Process and change given images
    """

    @staticmethod
    def negate(image):
        """
        Reverses every color and returns the negative of the given image
        """
        white = 255
        new_pixels = map(lambda x : list(map(lambda y : list(map(lambda \
        z : white - z, y)), x)), image.get_pixels())
        return RGBImage(new_pixels)

    @staticmethod
    def grayscale(image):
        """
        Converts the given image to black and white
        """
        pix = image.get_pixels()
        gray = RGBImage(pix)
        div = 3
        index = 2
        [[gray.set_pixel(i, j, (((pix[0][i][j] + pix[1][i][j] + \
        pix[index][i][j])/div), ((pix[0][i][j] + pix[1][i][j] + \
        pix[index][i][j])/div), ((pix[0][i][j] + pix[1][i][j] + \
        pix[index][i][j])/div) ))for j in range(len(pix[0][i]))] for i in \
        range(len(pix[0]))]
        return gray
    @staticmethod
    def clear_channel(image, channel):
        """
        Clears every value in the given channel
        """
        empty_pixels = list(map(lambda x : list(map(lambda y : 0, x)), \
        image.get_pixels()[channel]))
        output_image = image.get_pixels()
        output_image[channel] = empty_pixels
        return RGBImage(output_image)

    @staticmethod
    def crop(image, tl_row, tl_col, target_size):
        """
        Returns a copy of image reduced to passed in dimensions
        """
        br_row = 0
        if tl_row + target_size[0] >= image.size()[0]:
            br_row = image.size()[0]
        else:
            br_row = tl_row + target_size[0]

        br_col = 0
        if tl_col + target_size[1] >= image.size()[1]:
            br_col = image.size()[1]
        else:
            br_col = tl_col + target_size[1]

        output = RGBImage([[[image.pixels[let][i][j] for j in \
        range(tl_col, br_col)] for i in range(tl_row, br_row)] for \
        let in range(len(image.get_pixels()))])

        return output



    @staticmethod
    def chroma_key(chroma_image, background_image, color):
        """
        Replaces all pixels of a given color to the color of the
        background image at that pixel.
        """
        pix = chroma_image.get_pixels()
        output_image = RGBImage(pix)
        index = 2
        for i in range(len(pix[0])):
            for j in range(len(pix[0][i])):
                if pix[0][i][j] == color[0]:
                    if (pix[1][i][j] == color[1]) & (pix[index][i][j] == \
                    color[index]):
                        output_image.set_pixel(i, j, \
                        background_image.get_pixel(i, j))
        return output_image

    # rotate_180 IS FOR EXTRA CREDIT (points undetermined)
    @staticmethod
    def rotate_180(image):
        """
        Extra Credit
        """


# Part 3: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Implements a KNN classifier for given RGB images
    """

    def __init__(self, n_neighbors):
        """
        Initializes ImageKNNClassifier instance and n_neighbors
        """
        self.n_neighbors = n_neighbors
        self.data = None

    def fit(self, data):
        """
        Stores the training data
        """
        assert len(data) > self.n_neighbors
        assert self.data is None
        self.data = data

    @staticmethod
    def distance(image1, image2):
        """
        Calculates Euclidean distance between images for the value at
        each position.
        """
        assert image1.size() == image2.size()

        Euc = (lambda x : [a for z in x for y in z for a in y])

        img_1 = image1.get_pixels()
        elem1 = Euc(img_1)
        img_2 = image2.get_pixels()
        elem2 = Euc(img_2)
        sq = 2
        items = [(elem1[i] - elem2[i]) ** sq for i in range(len(elem1))]
        sq_root = 1/2
        dist = (sum(items)) ** sq_root
        return dist

    @staticmethod
    def vote(candidates):
        """
        Finds the most popular label in the given list of candidates
        """
        counter = {}
        for candidate in candidates:
            if candidate in counter:
                counter[candidate] += 1
            else:
                counter[candidate] = 1
        return max(counter, key = counter.get)


    def predict(self, image):
        """
        Predicts the label of a given image using KNN
        """
        dist = [(self.distance(image, i[0]), i[1]) for i in self.data]
        dist.sort(key = lambda x : x[0])
        neighbors = dist[:self.n_neighbors]
        prediction = [neighbor[1] for neighbor in neighbors]

        return self.vote(prediction)
