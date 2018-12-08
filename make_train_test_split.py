from random import shuffle
import csv
import glob

action_classes = ['noFights', 'fights']


def create_csvs():
    print("in create csvs")
    train = []
    test = []

    for myclass, directory in enumerate(action_classes):
        for filename in glob.glob('../Dataset/{}/*.avi'.format(directory)):
            print("filename %s" % filename)
            train.append([filename, myclass, directory])

    shuffle(train)
    shuffle(test)
    # print('train', len(total_train))
    # print('test', len(total_test))

    with open('train.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(['path', 'class', 'sport'])
        mywriter.writerows(train)
        print('Training CSV file created successfully')

    with open('test.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(['path', 'class', 'sport'])
        mywriter.writerows(test)
        print('Testing CSV file created successfully')

    print('CSV files created successfully')


if __name__ == "__main__":
    create_csvs()
