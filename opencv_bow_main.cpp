//BOW with SVM. Working example v1
//Based on https://github.com/mlefebvre/OpenHGR
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace cv;
using namespace cv::ml;

#define TRAIN_DATA_DIR "C:\\..."
#define EVAL_DATA_DIR "C:\\..."
#define FULL_FRAME_SIZE 0

Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<Feature2D> extractor = xfeatures2d::SURF::create();

int dictionary_size = 200;
TermCriteria criteria(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;

BOWKMeansTrainer bowTrainer(dictionary_size, criteria, retries, flags);
BOWImgDescriptorExtractor bowDE(extractor, matcher);

void extractTrainingVocabulary(const path& basepath) {
	Ptr<Feature2D> local_detector = xfeatures2d::SURF::create();

	for (directory_iterator iter = directory_iterator(basepath); iter != directory_iterator(); iter++) {
		directory_entry entry = *iter;

		if (is_directory(entry.path())) {
			std::cout << "Processing directory " << entry.path().string() << endl;
			extractTrainingVocabulary(entry.path());
		} else {
			path entryPath = entry.path();
			if (entryPath.extension() == ".jpg") {

				std::cout << "Working with " << entryPath.string() << endl;

				Mat image = imread(entryPath.string(), CV_LOAD_IMAGE_GRAYSCALE);

				#if FULL_FRAME_SIZE != 1
				cv::resize(image, image, Size(720, 480));
				#endif

				if (!image.empty()) {
					vector<KeyPoint> keypoints;
					local_detector->detect(image, keypoints);

					if (keypoints.empty()) {
						std::cout << "Keypoints is empty -> " << entryPath.string() << endl;
					} else {
						Mat features;

						local_detector->compute(image, keypoints, features);
						bowTrainer.add(features);

						features.release();
					}
				} else {
					std::cout << "Image reading error -> " << entryPath.string() << endl;
				}

				image.release();
			}
		}
	}
}

void extractBOWDescriptor(const path& basepath, Mat& descriptors, Mat& labels) {
	Ptr<Feature2D> local_detector = xfeatures2d::SURF::create();

	for (directory_iterator iter = directory_iterator(basepath); iter != directory_iterator(); iter++) {
		directory_entry entry = *iter;
		if (is_directory(entry.path())) {
			std::cout << "Processing directory " << entry.path().string() << endl;
			extractBOWDescriptor(entry.path(), descriptors, labels);
		} else {
			path entryPath = entry.path();
			if (entryPath.extension() == ".jpg") {
				std::cout << "Working with " << entryPath.string() << endl;

				Mat image = imread(entryPath.string(), CV_LOAD_IMAGE_GRAYSCALE);

				#if FULL_FRAME_SIZE != 1
				cv::resize(image, image, Size(720, 480));
				#endif

				if (!image.empty()) {
					vector<KeyPoint> keypoints;
					local_detector->detect(image, keypoints);

					if (keypoints.empty()) {
						std::cout << "Keypoints is empty -> " << entryPath.string() << endl;
					} else {
						Mat bowDescriptor;

						bowDE.compute(image, keypoints, bowDescriptor);
						descriptors.push_back(bowDescriptor);
						float label = atof(entryPath.filename().string().c_str());
						std::cout << "Label: " << (int)label << endl;
						labels.push_back((int)label);

						bowDescriptor.release();
					}
				} else {
					std::cout << "Image reading error -> " << entryPath.string() << endl;
				}

				image.release();
			}
		}
	}
}

int main() {
	double time_dictionary_build, time_eval_build;
	Mat trainData(0, dictionary_size, CV_32FC1);
	Mat trainLabels(0, 1, CV_32S);

	clock_t tStart = clock();
	std::cout << "Creating dictionary..." << endl;
	extractTrainingVocabulary(path(TRAIN_DATA_DIR));
	std::cout << "Clustering features..." << endl;
	Mat dictionary = bowTrainer.cluster();
	bowDE.setVocabulary(dictionary);

	std::cout << "Processing training data..." << endl;
	extractBOWDescriptor(path(TRAIN_DATA_DIR), trainData, trainLabels);

	time_dictionary_build = (double)(clock() - tStart) / CLOCKS_PER_SEC;
	printf("Dictionary build time: %.2fs\n", time_dictionary_build);

	Ptr<SVM> svm = SVM::create();
	svm->setKernel(SVM::CHI2);
	svm->setType(SVM::C_SVC);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	std::cout << "Training classifier.... using SVM" << endl;
	svm->train(trainData, ROW_SAMPLE, trainLabels);

	tStart = clock();
	Mat evalData(0, dictionary_size, CV_32FC1);
	Mat realLabels(0, 1, CV_32FC1);
	extractBOWDescriptor(path(EVAL_DATA_DIR), evalData, realLabels);
	time_eval_build = (double)(clock() - tStart) / CLOCKS_PER_SEC;

	Mat predictedLabels(1, 1, CV_32FC1);
	svm->predict(evalData, predictedLabels);

	int error_count = 0;
	for (int i = 0; i < realLabels.rows; i++) {
		if (realLabels.at<int>(i, 0) != predictedLabels.at<float>(i, 0)) error_count++;
	}

	printf("SVM error rate: %.2fs\n", ((double)error_count / realLabels.rows) * 100);
	printf("Dictionary build time: %.2fs\n", time_dictionary_build);
	printf("Eval build time: %.2fs\n", time_eval_build);

	system("pause");

	return 0;
}
