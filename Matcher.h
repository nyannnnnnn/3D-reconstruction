#include <iostream> 
#include <string> 
#include <iomanip>  
#include <sstream> 

#include "opencv/cv.h"

using namespace std;
using namespace cv;
using namespace xfeatures2d;
class Matcher {
private:	
	// pointer to the feature point detector object
	cv::Ptr<SURF> detector;
	// pointer to the feature descriptor extractor object
	cv::Ptr<SURF> extractor;
	float ratio; // max ratio between 1st and 2nd NN
	bool refineF; // if true will refine the F matrix
	double distance; // min distance to epipolar
	double confidence; // confidence level (probability)
	Mat fundamentalMatrics;
public:
	Matcher() ;

	Mat getFundamentalMatrix();

	void setFeatureDetector(Ptr<SURF> &detect);

	void setDescriptorExtractor(Ptr<SURF> & desExtract);

	void setMinDistanceToEpipolar(double distance);

	void setRatio(float ratio);

	bool match (Mat &image1, Mat &image2, vector<DMatch> &matches, vector<KeyPoint> &keypoints1,vector<KeyPoint> &keypoints2);

	int ratioTest(std::vector<std::vector<cv::DMatch> > &matches);

	void symmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1,	const std::vector<std::vector<cv::DMatch> >& matches2, std::vector<cv::DMatch>& symMatches);

	void setConfidenceLevel(double confidence);

	cv::Mat ransacTest(
		const std::vector<cv::DMatch>& matches,
		const std::vector<cv::KeyPoint>& keypoints1,
		const std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::DMatch>& outMatches);


};

