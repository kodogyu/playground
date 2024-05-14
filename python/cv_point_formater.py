import csv

if __name__ == "__main__":
    read_file_name = "files/frames_feature_info.csv"
    write_file_name = "files/cv_feature_format.txt"

    num_feature = 20

    with open(write_file_name, 'w') as write_file:
        # csv_writer = csv.writer(write_file)

        with open(read_file_name, 'r') as read_file:
            csv_reader = csv.reader(read_file)

            is_first = True
            for row in csv_reader:
                cv_format = []

                # skip header
                if is_first:
                    is_first = False
                    continue

                for elem in row[:num_feature]:
                    # make tuple
                    try:
                        first, second = elem.split(sep=', ')
                    except ValueError:
                        print("value error.")
                        continue
                    finally:
                        first = first[1:]
                        second = second[:-1]

                    # float tuple
                    float_tuple_elem = (float(first), float(second))
                    # int tuple
                    int_tuple_elem = (round(float(first)), round(float(second)))
                    # check
                    print(float_tuple_elem, int_tuple_elem)

                    # append to list
                    # cv_format.append(int_tuple_elem)
                    cv_format.append(float_tuple_elem)

                # make it to cv Point format
                cv_format = ["cv::Point2f" + str(elem) for elem in cv_format]
                # write data
                write_file.write(', '.join(cv_format) + '\n')