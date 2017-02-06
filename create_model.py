
from optparse import OptionParser
import logging
import sys

def read_images(path, image_size=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X, y, folder_names]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
            folder_names: The names of the folder, so you can display it in a prediction.
    """
    c = 0
    X = []
    y = []
    folder_names = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    if (image_size is not None):
                        im = cv2.resize(im, image_size)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y,folder_names]


def validateModel(model, numfolds, images, labels):

    print "Validating model with %s folds..." % numfolds
    # We want to have some log output, so set up a new logging handler
    # and point it to stdout:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add a handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Perform the validation & print results:
    crossval = KFoldCrossValidation(model, k=numfolds)
    crossval.validate(images, labels)
    crossval.print_results()


if __name__ == '__main__':
    # model.pkl is a pickled (hopefully trained) PredictableModel, which is
    # used to make predictions. You can learn a model yourself by passing the
    # parameter -d (or --dataset) to learn the model from a given dataset.
    usage = "usage: %prog [options] model_filename"
    # Add options for training, resizing, validation and setting the camera id:
    parser = OptionParser(usage=usage)
    parser.add_option("-r", "--resize", action="store", type="string", dest="size", default="100x100", 
        help="Resizes the given dataset to a given size in format [width]x[height] (default: 100x100).")
    parser.add_option("-v", "--validate", action="store", dest="numfolds", type="int", default=None, 
        help="Performs a k-fold cross validation on the dataset, if given (default: None).")
    parser.add_option("-t", "--train", action="store", dest="dataset", type="string", default=None,
        help="Trains the model on the given dataset.")
    parser.add_option("-i", "--id", action="store", dest="camera_id", type="int", default=0)

    # Parse arguments
    (options, args) = parser.parse_args()
   
    # Check model 
    if len(args) == 0:
        parser.print_help()
        sys.exit()

    # Check dataset
    if not options.dataset:
        print "[Error] A dataset is needed!"
        sys.exit()

    # Resize the images, as this is neccessary for some algorithms
    # Default is 100x100
    # Note LBPH doesn't hve this requirement
    try:
        image_size = (int(options.size.split("x")[0]), int(options.size.split("x")[1]))
    except:
        print "[Error] Unable to parse the given image size '%s'. Please pass it in the format [width]x[height]!" % options.size
        sys.exit()
   

    print "size : " + str(image_size)

    model_filename = args[0]


    # Check if the given dataset exists:
    if not os.path.exists(options.dataset):
        print "[Error] No dataset found at '%s'." % dataset_path
        sys.exit()    
    

    # Reads the images, labels and folder_names from a given dataset. Images
    # are resized to given size on the fly:
    print "Loading dataset..."
    [images, labels, subject_names] = read_images(options.dataset, image_size)
    # Zip us a {label, name} dict from the given data:
    list_of_labels = list(xrange(max(labels)+1))
    subject_dictionary = dict(zip(list_of_labels, subject_names))
    # Get the model we want to compute:
    model = get_model(image_size=image_size, subject_names=subject_dictionary)
    # Sometimes you want to know how good the model may perform on the data
    # given, the script allows you to perform a k-fold Cross Validation before
    # the Detection & Recognition part starts:
  
    # Validate model
    if options.numfolds:
        validateModel(model, options.numfolds, images, labels)
        sys.exit() 
    
    
    # Compute the model:
    print "Computing the model..."
    model.compute(images, labels)
    # And save the model, which uses Pythons pickle module:
    print "Saving the model..."
    save_model(model_filename, model)
else:
    print "Loading the model..."
    model = load_model(model_filename)
        
