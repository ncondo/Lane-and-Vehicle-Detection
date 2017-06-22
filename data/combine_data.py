import os
import glob


def combine_data():
    """
    Read in image file names from all vehicle and non-vehicles subfolders and
    save as cars.txt and not-cars.txt files.
    """
    basedir = 'vehicles/'
    # Different folders represent different sources for images e.g. GTI, KITTI
    sub_dirs = os.listdir(basedir)
    cars = []
    for sub_dir in sub_dirs:
        cars.extend(glob.glob(basedir+sub_dir+'/*'))

    print('Number of vehicle images found: ', len(cars))
    # Save all vehicle file names to cars.txt
    with open('cars.txt', 'w') as f:
        for fname in cars:
            f.write('data/'+fname+'\n')
    f.close()

    # Do the same for non-vehicle images
    baseir = 'non-vehicles/'
    sub_dirs = os.listdir(basedir)
    not_cars = []
    for sub_dir in sub_dirs:
        not_cars.extend(glob.glob(basedir+sub_dir+'/*'))

    print('Number of non-vehicle images found: ', len(not_cars))
    # Save all non-vehicle file names to not_cars.txt
    with open('not_cars.txt', 'w') as f:
        for fname in not_cars:
            f.write('data/'+fname+'\n')
    f.close()


if __name__=='__main__':
    combine_data()
