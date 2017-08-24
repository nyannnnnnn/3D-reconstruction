//
//  main.cpp
//  3d
//
//  Created by my Mac on 2017/4/1.
//  Copyright © 2017年 my Mac. All rights reserved.


#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <math.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv/cv.h>

#include "dataTypeStructs.h"
#include "matcher.h"
#include "MatchingPoi.h"
#include "MatrixCalc.h"
#include "buildModel.h"
#include "PCT.h"

using namespace std;
using namespace cv;
using namespace xfeatures2d;

void nearest(Mat object, Mat image);
void downsample(Mat *image);
void DrawEpiLines(const Mat& img_1, const Mat& img_2, vector<Point2f>points1, vector<Point2f>points2);
void doSIFT(Mat);
void doSURF(Mat);
vector<KeyPoint> *robustMatching(Mat image1, Mat image2);
Matx34d tableProcess(Matx34d P1,
                     vector<KeyPoint> newKeyPoints,
                     vector<KeyPoint> oldKeyPoints,
                     PCTable *current,
                     PCTable *previous, Mat K);

int fileNumber;

int main(int argc, char *argv[], char **window_name) {
    
    fileNumber = 0;
    
    const string image = argv[1];
    assert(!image.empty());
    const string arg2 = argv[2];
    assert(!arg2.empty());
    const string extension = ".png";
    int numberPictures = atoi(arg2.c_str());
    
    int pictureNumber1 = 0;
    int pictureNumber2 = 1;
    
    
    string imageName1 = image + "0000" + extension;
    string imageName2 = image + "0001" + extension;
    
    
    Mat frame1 = imread(imageName1, -1);
    Mat frame2 = imread(imageName2, -1);
    
    
    MatrixCalc matrixCalculator;
    buildModel buildModel;
    
    vector<SpacePoint> pointCloud;
    Matx34d P;
    Matx34d P1;
    
    int previousNumberOfPointsAdded = 0;
    
    bool initial3dModel = true;
    
    PCTable table1;
    PCTable table2;
    
    table2.init();
    table1.init();
    
    PCTable *current = &table1;
    PCTable *previous = &table2;
    
    int factor = 1;
    int count = 0;
    
    while (fileNumber < numberPictures - 1)
    {
        
        cout << endl << endl << endl << "Using " << imageName1 << " and " << imageName2 << endl;
        namedWindow("Frame 1", WINDOW_NORMAL);
        moveWindow("Frame 1", 50, 80);
        imshow("Frame 1", frame1);
        
        namedWindow("Frame 2", WINDOW_NORMAL);
        moveWindow("Frame 2", 400, 80);
        imshow("Frame 2", frame2);
        doSIFT(frame1);
        doSURF(frame1);
        doSIFT(frame2);
        doSURF(frame2);
        cout << "Matching..." << endl;
        MatchingPoints robustMatcher(frame1, frame2);
        vector<KeyPoint> keypoints1 = robustMatcher.getKeyPoints1();
        vector<KeyPoint> keypoints2 = robustMatcher.getKeyPoints2();
        vector<Colours> keyPointColours = robustMatcher.getColours(frame1);
        
        
        if (robustMatcher.hasEnoughMatches())
        {
            vector<Point2f> p1, p2;
            for(int i = 15; i < 30; i++){
                p1.push_back(keypoints1[i].pt);
                p2.push_back(keypoints2[i].pt);
            }
            DrawEpiLines(frame1, frame2, p1, p2);
            robustMatcher.displayFull(frame1, frame2);
            cout << "Enough Matches! " << endl;
            if (DEBUG == 0)
            {
                cout << endl << keypoints1.size() << " matches!" << endl;
            }
            //add into point cloud
            Mat K = matrixCalculator.findMatrixK(frame1);
            if (initial3dModel == true)
            {
                cout << "Calculating initial camera matricies..." << endl;
                matrixCalculator.FindCameraMatrices(keypoints1, keypoints2, robustMatcher.getFundamentalMatrix(), P, P1, pointCloud);
                
                cout << "Creating initial 3D model..." << endl;
                pointCloud = matrixCalculator.triangulation(keypoints1, keypoints2, K, P, P1, pointCloud);
                (*current).addAllEntries(keypoints2, pointCloud);
                
                cout << "Initial Lookup table size is: " << current->tableSize() << endl;
                initial3dModel = false;
            }
            else
            {
                cout << "Previous (current)  Table Size is " << current->tableSize() << endl;
                
                cout << "Previous (previous)  Table Size is " << previous->tableSize() << endl;
                
                
                previous->init();
                previous = current;
                
                if(current == & table2)
                {
                    current = &table1;
                    
                } else
                {
                    current = &table2;
                }
                
                cout << "LookupTable Size is: " << previous->tableSize() << endl;
                cout << "New Table Size is: " << current->tableSize() << endl;
                
                P = P1; //images get shuffled along
                P1 = tableProcess(P1, keypoints2, keypoints1, current, previous, K);
                
                cout << "New Table Size after adding known 3d Points: " << current->tableSize() << endl;
                
                cout << "Triangulating..." << endl;
                pointCloud = matrixCalculator.triangulation(keypoints1, keypoints2, K, P, P1, pointCloud);
                current->addAllEntries(keypoints2, pointCloud);
                
                cout << "Table Size after adding Triangulated points: " << current->tableSize() << endl;
                
            }
            
            int numberOfPointsAdded = static_cast<int>(keypoints1.size());
            cout << "Start writing points to file..." << endl;
            //create new file each time we process features from a new image
            buildModel.insert_header(static_cast<int>(pointCloud.size()), fileNumber);
            
            //write previous points;
            for (int i = 0; i < previousNumberOfPointsAdded; i++)
            {
                Point3d point = pointCloud.at(i).point;
                int blue = pointCloud.at(i).blue;
                int green = pointCloud.at(i).green;
                int red = pointCloud.at(i).red;
                buildModel.insert_point(point.x, point.y, point.z, red, green, blue, fileNumber);
            }
            
            //write current points
            for (int i = 0; i < numberOfPointsAdded; i++)
            {
                Point3d point = pointCloud.at(i + previousNumberOfPointsAdded).point;
                Colours pointColour = keyPointColours.at(i);
                pointCloud.at(i + previousNumberOfPointsAdded).blue = pointColour.blue;
                pointCloud.at(i + previousNumberOfPointsAdded).red = pointColour.red;
                pointCloud.at(i + previousNumberOfPointsAdded).green = pointColour.green;
                buildModel.insert_point(point.x, point.y, point.z, pointColour.red, pointColour.green, pointColour.blue, fileNumber);
            }
            
            fileNumber++;
            
            previousNumberOfPointsAdded = numberOfPointsAdded + previousNumberOfPointsAdded;
            
            cout << "End adding points" << endl;
            
        }
        else
        {
            cout << "Not enough matches!" << endl;
        }
        cout << imageName1 << " " << imageName2 << " done,  Image is " << frame1.cols << "x" << frame1.rows << " " <<endl;
        
        pictureNumber1 = (pictureNumber2) % numberPictures;
        pictureNumber2 = (pictureNumber2 + factor) % numberPictures;
        waitKey(20);
        
        count++;
        cout << "Count is " << count << " Factor is " << factor << endl;
        if (count % (numberPictures) == numberPictures - 1)
        {
            pictureNumber2++;
            factor++;
        }
        string stringpicturenumber1 = static_cast<ostringstream>((ostringstream() << pictureNumber1)).str();
        string stringpicturenumber2 = static_cast<ostringstream>((ostringstream() << pictureNumber2)).str();
        if(pictureNumber1 < 10){
            stringpicturenumber1 = "000" + stringpicturenumber1;
        }
        if(pictureNumber2 < 10){
            stringpicturenumber2 = "000" + stringpicturenumber2;
        }
        if(pictureNumber2 == 10){
            stringpicturenumber2 = "00" + stringpicturenumber2;
        }
        imageName1 = image + stringpicturenumber1 + extension;
        imageName2 = image + stringpicturenumber2 + extension;
        frame1 = imread(imageName1, -1);
        frame2 = imread(imageName2, -1);
    }
    
    cout << endl << "Done" << endl;
    char key = (char)waitKey(0);
    switch (key) {
        case 's':
            break;
        case 'd':
            pictureNumber1 = (pictureNumber1 + 1)%numberPictures;
            pictureNumber2 = (pictureNumber2 + 1)%numberPictures;
            break;
        case 'r':
            //do it again
            break;
        case 'q':
            return 0;
            break;
    }
    return 0;
}

Matx34d tableProcess(Matx34d P1,
                     vector<KeyPoint> newKeyPoints,
                     vector<KeyPoint> oldKeyPoints,
                     PCTable *current,
                     PCTable *previous,
                     Mat K)
{
    Point3d *found;
    vector<Point2d> foundPoints2d;
    vector<Point3d> foundPoints3d;
    vector<KeyPoint> newKeyPoints_notIn;
    vector<KeyPoint> oldKeyPoints_notIn;
    
    for (int i = 0; i < oldKeyPoints.size(); i++)
    {
        found = previous->find_3d(oldKeyPoints.at(i).pt);
        
        if (found != NULL)
        {
            Point3d newPoint;
            newPoint.x = found->x;
            newPoint.y = found->y;
            newPoint.z = found->z;
            Point2d newPoint2;
            newPoint2.x = newKeyPoints.at(i).pt.x;
            newPoint2.y = newKeyPoints.at(i).pt.y;
            foundPoints3d.push_back(newPoint);
            foundPoints2d.push_back(newPoint2);
            current->add_entry(&newPoint, &newPoint2);
        }
    }
    
    //cout << foundPoints3d.size();
    cout << "Matches found in table: " << foundPoints2d.size() << endl;
    
    int size = static_cast<int>(foundPoints3d.size());
    
    
    Mat_<double> found3dPoints(size, 3);
    Mat_<double> found2dPoints(size, 2);
    
    for (int i = 0; i < size; i++)
    {
        
        found3dPoints(i, 0) = foundPoints3d.at(i).x;
        found3dPoints(i, 1) = foundPoints3d.at(i).y;
        found3dPoints(i, 2) = foundPoints3d.at(i).z;
        
        found2dPoints(i, 0) = foundPoints2d.at(i).x;
        found2dPoints(i, 1) = foundPoints2d.at(i).y;
        
    }
    
    Mat_<double> temp1(found3dPoints);
    Mat_<double> temp2(found2dPoints);
    
    Mat P(P1);
    
    Mat r(P, Rect(0, 0, 3, 3));
    Mat t(P, Rect(3, 0, 1, 3));
    
    Mat r_rog;
    cv::Rodrigues(r, r_rog);
    
    
    Mat dist = Mat::zeros(1, 4, CV_32F);
    double _dc[] = {0, 0, 0, 0};
    
    cv::solvePnP(found3dPoints, found2dPoints, K, Mat(1, 4, CV_64FC1, _dc), r_rog, t, false);
    
    cout << "Got new Camera matrix" << endl;
    
    Mat_<double> R1(3, 3);
    Mat_<double> t1(t);
    
    cv::Rodrigues(r_rog, R1);
    
    Mat camera = (Mat_<double> (3,4) << 	R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
                  R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
                  R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
    
    return Matx34d(camera);
}


void nearest(Mat image1, Mat image2)
{
    int minHessian = 1000;
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keypoints1, keypoints2;
    
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);
    
    Ptr<SURF> extractor = SURF::create();
    
    Mat descriptors1, descriptors2;
    
    extractor->compute(image1, keypoints1, descriptors1);
    extractor->compute(image2, keypoints2, descriptors2);
    
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    int matchAmount = 200;
    
    std::nth_element(matches.begin(), matches.begin() + matchAmount, matches.end());
    matches.erase(matches.begin() + matchAmount, matches.end());
    
    
    Mat imageMatches;
    drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches);
    namedWindow("Compare image", WINDOW_NORMAL);
    imshow("Compare image", imageMatches);
}


void DrawEpiLines(const Mat& img_1, const Mat& img_2, vector<Point2f>points1, vector<Point2f>points2){
    
    cv::Mat F = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    
    std::vector<cv::Vec<float, 3> > epilines1, epilines2;
    cv::computeCorrespondEpilines(points1, 1, F, epilines1);//compute epilines ax+by+c=0;
    cv::computeCorrespondEpilines(points2, 2, F, epilines2);
    //convert to RGB
    cv::Mat img1, img2;
    if (img_1.type() == CV_8UC3)
    {
        img_1.copyTo(img1);
        img_2.copyTo(img2);
    }
    else if (img_1.type() == CV_8UC1)
    {
        cvtColor(img_1, img1, COLOR_GRAY2BGR);
        cvtColor(img_2, img2, COLOR_GRAY2BGR);
    }
    else
    {
        cout << "unknow img type\n" << endl;
        exit(0);
    }
    
    cv::RNG& rng = theRNG();
    for (uint i = 0; i < points2.size(); i++)
    {
        Scalar color = Scalar(rng(256), rng(256), rng(256));//random color
        
        circle(img2, points2[i], 5, color);//circle the feature points
        line(img2, Point(0, -epilines1[i][2] / epilines1[i][1]), Point(img2.cols, -(epilines1[i][2] + epilines1[i][0] * img2.cols) / epilines1[i][1]), color);
        circle(img1, points1[i], 4, color);
        line(img1, cv::Point(0, -epilines2[i][2] / epilines2[i][1]), cv::Point(img1.cols, -(epilines2[i][2] + epilines2[i][0] * img1.cols) / epilines2[i][1]), color);
        
    }
    cv::imshow("img2 epiline1", img2);
    imwrite("/Users/nyan/Downloads/line1.jpg", img2);
    cv::imshow("img1 epiline2", img1);
    imwrite("/Users/nyan/Downloads/line2.jpg", img1);
    
    waitKey(20);
}
void doSIFT(Mat img){
    
    vector<KeyPoint> key_points;    //feature points
    Ptr<FeatureDetector> detector = cv::KAZE::create( "SIFT" );//generate SIFT detector
    Ptr<DescriptorExtractor> descriptor_extractor = cv::KAZE::create( "SIFT" );    Mat descriptors, mascara;
    Mat output_img;    //output matrix
    detector->detect( img, key_points );
    descriptor_extractor->compute( img, key_points, descriptors );
    //draw feature points
    drawKeypoints(img,
                  key_points,
                  output_img,
                  Scalar::all(-1),
                  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    namedWindow("SIFT");
    imshow("SIFT", output_img);
    waitKey(20);
}
void doSURF(Mat img){
    int minHessian = 700;//hessian threshold
    Mat output_img;
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keyPoint;
    

    detector->detect(img, keyPoint);
    
    Ptr<SURF> extractor = SURF::create();
    Mat descriptors;
    extractor->compute(img, keyPoint, descriptors);
    drawKeypoints(img,
                  keyPoint,
                  output_img,
                  Scalar::all(-1),
                  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    namedWindow("SURF");
    imshow("SURF", output_img);
    waitKey(20);
}
