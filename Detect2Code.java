import org.opencv.core.Core;
import org.opencv.core.Size;
import org.opencv.core.CvType;
import org.opencv.core.Scalar;
import org.opencv.core.Mat;
import org.opencv.core.RotatedRect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;


import java.util.ArrayList;

public class Detect2Code {
 
    public static void main(String[] args) {
        System.out.println("Welcome to OpenCV " + Core.VERSION);
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
//      int radius = 3;
//    	int topMargin = 5;
//    	int leftMargin = 5;
//    	int dotMargin = 2;
//    	int a[] = { 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0 };			//code		第五位开始识别,最好第六位设置为1
//    	int code_length = 11;
//	
//    	String address = "./other.jpg";
    	//doting(address, radius, topMargin, leftMargin, dotMargin, a,code_length);
    	
    	String recognition_address = "./dot_test2.jpg";
//    	Mat image_original = Imgcodecs.imread(recognition_address);
//    	Mat rotation_final = new Mat();
//    	Mat M = Imgproc.getRotationMatrix2D(new Point(image_original.rows()/2,image_original.cols()/2), 95, 1.0);
//    	Imgproc.warpAffine(image_original, rotation_final, M, image_original.size());     
//    	Imgcodecs.imwrite("test2.jpg", rotation_final);
    	int code = image2code(recognition_address);
//    	System.out.println(code);
    }
    //转置一下
    static void reverse(Mat src,Mat dst)
    {
    	for(int i = 0 ; i < src.rows(); i ++) {
    		for(int k = 0 ; k < src.cols(); k ++) {
    			if(src.get(i,k)[0] < 40)
    				dst.put(i, k, 255);
    			else 
    				dst.put(i, k, 0);
    		}
    	}
    }
    
    public static int image2code(String address) {
    	//现在的方法稳定性不够
    	int code = 0;
    	
    	// image graying
    	Mat image_original= Imgcodecs.imread(address);
    	Mat image_gray = new Mat(image_original.size(),CvType.CV_8U,new Scalar(0));    	
    	Imgproc.cvtColor(image_original, image_gray, Imgproc.COLOR_BGR2GRAY);
    	
    	// edge detection (use canny operator)
    	int threshold = 60;
    	Mat image_canny = new Mat(image_gray.size(), CvType.CV_8U, new Scalar(0));
    	Imgproc.blur(image_gray, image_gray, new Size(3,3));
    	Imgproc.Canny(image_gray, image_canny, threshold, threshold + 30);
    	// other threshold algorithm which can get the target region.	
    	//Imgproc.threshold(image_gray, image_canny, 120, 255, Imgproc.THRESH_OTSU);		// 如果选择的是OTSU算法的话，阈值是多少都是无所谓的
    	//Imgproc.adaptiveThreshold(image_gray, image_canny, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 5, 4);  	
		//int scale = 1;  
		//int delta = 0;  
		//int ddepth = CvType.CV_8U;  
		//int kernel_size = 4;   
		//Imgproc.Laplacian( image_gray, image_canny, ddepth, kernel_size, scale, delta,Core.BORDER_DEFAULT );  
	
    	// morphology operation
		//Imgproc.morphologyEx(img_canny, img_canny, Imgproc.MORPH_CLOSE, new Mat(3,3,CvType.CV_8U));	
    	//Mat ele_of_threshold = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(31,31));
    	//Imgproc.dilate(image_canny, mor_after_threshold, ele_of_threshold);
    	//Imgproc.morphologyEx(src, dst, op, kernel);
    	//Imgproc.erode(img_canny, img_canny, ele_out);
  		
    	// Make contours of target region	
    	ArrayList<MatOfPoint> target_mask_contours = new ArrayList<MatOfPoint>();		//find the contours of images
    	Mat hierarchy_of_target_mask = new Mat();
    	hierarchy_of_target_mask.convertTo(hierarchy_of_target_mask, CvType.CV_8U);
    	Imgproc.findContours(image_canny, target_mask_contours,hierarchy_of_target_mask, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);	
    	int max_index = 0;
    	double max_area = 0;
    	for (int i = 0; i < (int)target_mask_contours.size(); i++)
    	{
    		double g_dConArea = Imgproc.contourArea(target_mask_contours.get(i));
    		//System.out.println(g_dConArea);
    		if (g_dConArea >= max_area) {
    			max_area = g_dConArea;    			
    			max_index = i;
    		}
    	}
    	Mat target_mask = new Mat(image_canny.size(), CvType.CV_8U,new Scalar(0));
    	Imgproc.drawContours(target_mask, target_mask_contours, max_index,new Scalar(255), -1);	// target_mask is the target region
    	   	
        MatOfPoint2f mp2f =new MatOfPoint2f( target_mask_contours.get(max_index).toArray());
        RotatedRect rect = Imgproc.minAreaRect(mp2f);
        System.out.println(rect.angle);
        Point[] P = new Point[4];
        rect.points(P);
        
        // make the min area of rect is right for us
        double angle_rect = rect.angle;
        Mat rotation_mid = new Mat();
        
        if(angle_rect >=0.0) {
        	Mat M = Imgproc.getRotationMatrix2D(new Point(image_original.rows()/2,image_original.cols()/2), -angle_rect, 1.0);
        	Imgproc.warpAffine(image_gray, rotation_mid, M, image_original.size());
        }else {
        	Mat M = Imgproc.getRotationMatrix2D(new Point(image_original.rows()/2,image_original.cols()/2), 90+angle_rect, 1.0);
        	Imgproc.warpAffine(image_gray, rotation_mid, M, image_original.size());     	
        }
        
        // Detect the circle in the cornor of image(OpenCV only have Hough Circle. So I use this algorithm2-1)
    	Mat circles = new Mat();   	
    	int iCannyUpperThreshold = 90;
    	int iMinRadius = 15;
    	int iMaxRadius = 60;
    	int iAccumulator = 120;
    	
    	//Imgproc.blur(threshold_image, threshold_image, new Size(3,3));
    	Imgproc.HoughCircles(rotation_mid, circles, Imgproc.CV_HOUGH_GRADIENT, 
    	         4.0, image_gray.rows()*2 /3, iCannyUpperThreshold, iAccumulator, 
    	         iMinRadius, iMaxRadius);
    	//Circle Test Code:
    	ArrayList<Point> points_circle = new ArrayList<Point>();
    	if (circles.cols() > 0){
    	    for (int x = 0; x < circles.cols(); x++) {
    	        double vCircle[] = circles.get(0,x);
    	        if (vCircle == null)
    	            break;
    	        Point pt = new Point(Math.round(vCircle[0]), Math.round(vCircle[1]));
 //   	        int radius = (int)Math.round(vCircle[2]);
    	        points_circle.add(pt);
    	        // draw the found circle
//    	        Imgproc.circle(image_gray, pt, radius, new Scalar(255),3);
//    	        Imgproc.circle(image_gray, pt, 3, new Scalar(255), -1);
//    	        System.out.println(Math.round(vCircle[0]) + "  " +  Math.round(vCircle[1]) + "  " +  Math.round(vCircle[2]));
    	    }
    	}
//
    	double rot_angle2 = 0.0;
    	// judge the keypoint
    	for(int i = 0 ;i < points_circle.size(); i ++) {
    		
    		if(image_gray.get((int)points_circle.get(i).x, (int)points_circle.get(i).y - 11)[0] > 125) {	//向上移动几个像素点，看效果
    			if (points_circle.get(i).x > image_gray.rows()/2 && points_circle.get(i).y > image_gray.cols()/2) {
    				rot_angle2 = 0;
    			}
    			else if (points_circle.get(i).x < image_gray.rows()/2 && points_circle.get(i).y > image_gray.cols()/2) {
    				rot_angle2 = 90;
    			}
    			else if (points_circle.get(i).x > image_gray.rows()/2 && points_circle.get(i).y < image_gray.cols()/2) {
    				rot_angle2 = 180;
    			}else {
    				rot_angle2 = 270;
    			}
    		}
    	}
    	Mat rotation_final = new Mat();
    	Mat M = Imgproc.getRotationMatrix2D(new Point(image_original.rows()/2,image_original.cols()/2), rot_angle2, 1.0);
    	Imgproc.warpAffine(rotation_mid, rotation_final, M, image_original.size());     
    	Imgcodecs.imwrite("test2.jpg", rotation_final);
//        
//        // 计算外接矩形的长和宽
//        int out_rect_height = Math.abs((int)(P[0].y - P[1].y));
//  		int out_rect_width  = Math.abs((int)(P[2].x - P[1].x));
        

//    	Imgproc.Canny(image_gray, threshold_image, 60, 90);
//    	
//    	Mat circles = new Mat();   	
//		Imgproc.adaptiveThreshold(image_gray, mask_of_foreground, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 3, 5);
//    	int iCannyUpperThreshold = 90;
//    	int iMinRadius = 15;
//    	int iMaxRadius = 25;
//    	int iAccumulator = 100;
//    	
//    	//Imgproc.blur(threshold_image, threshold_image, new Size(3,3));
//    	Imgproc.GaussianBlur(threshold_image, threshold_image, new Size(3,3), 0);
//    	Imgproc.HoughCircles(threshold_image, circles, Imgproc.CV_HOUGH_GRADIENT, 
//    	         2.0, threshold_image.rows() /2, iCannyUpperThreshold, iAccumulator, 
//    	         iMinRadius, iMaxRadius);
//    	//Imgproc.HoughCircles(image, circles, method, dp, minDist, param1, param2, minRadius, maxRadius);
//    	ArrayList<Point> points_circle = new ArrayList<Point>();
//    	if (circles.cols() > 0)
//    	    for (int x = 0; x < circles.cols(); x++) 
//    	        {
//    	        double vCircle[] = circles.get(0,x);
//
//    	        if (vCircle == null)
//    	            break;
//
//    	        Point pt = new Point(Math.round(vCircle[0]), Math.round(vCircle[1]));
//    	        int radius = (int)Math.round(vCircle[2]);
//    	        points_circle.add(pt);
//    	        // draw the found circle
//    	        Imgproc.circle(image_gray, pt, radius, new Scalar(255),3);
//    	        Imgproc.circle(image_gray, pt, 3, new Scalar(255), -1);
//    	        }
//
//    	Imgcodecs.imwrite("test2.jpg", image_gray);
//    	int threshold = 60;
//    	Mat img_canny = new Mat(image_gray.size(), CvType.CV_8U, new Scalar(0));
//    	Imgproc.Canny(image_gray, img_canny, threshold, threshold + 30);
//    	
//    	Imgcodecs.imwrite("test2.png", img_canny);
//    	ArrayList<MatOfPoint> target_mask_contours = new ArrayList<MatOfPoint>();		//find the contours of images
//
//    	Mat hierarchy_target_mask = new Mat();
//    	hierarchy_target_mask.convertTo(hierarchy_target_mask, CvType.CV_8U);
//    	Imgproc.findContours(img_canny, target_mask_contours,hierarchy_target_mask, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
//    	
//    	int max_index = 0;
//    	double max_area = 0;
//    	for (int i = 0; i < (int)target_mask_contours.size(); i++)
//    	{
//    		double g_dConArea = Imgproc.contourArea(target_mask_contours.get(i));
//    		//System.out.println(g_dConArea);
//    		if (g_dConArea >= max_area) {
//    			max_area = g_dConArea;
//    			
//    			max_index = i;
//    		}
//    	}
//    	
//    	Mat target_mask = new Mat(img_canny.size(), CvType.CV_8U,new Scalar(0));
//    	Imgproc.drawContours(target_mask, target_mask_contours, max_index,new Scalar(255), -1);	//中间区域的mask
//    	
//    	
//    	ArrayList<MatOfPoint> contours_out = new ArrayList<MatOfPoint>();
//    	Mat hierarchy_out = new Mat();
//        hierarchy_out.convertTo(hierarchy_out, CvType.CV_8U);
//        Imgproc.findContours(target_mask, contours_out, hierarchy_out, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
//    	
//        MatOfPoint2f mp2f =new MatOfPoint2f( contours_out.get(0).toArray());
//        RotatedRect rect = Imgproc.minAreaRect(mp2f);
//        System.out.println(rect.angle);
//        Point[] P = new Point[4];
//        rect.points(P);
//        
//        int out_rect_height = Math.abs((int)(P[0].y - P[1].y));
//    	int out_rect_width  = Math.abs((int)(P[2].x - P[1].x));
    	// 控制转向
//    	Mat dst = new Mat();
//    	Mat M = Imgproc.getRotationMatrix2D(new Point(image_original.rows()/2,image_original.cols()/2), 5, 1.0);
//    	Imgproc.warpAffine(image_original, dst, M, image_original.size());
//    	Imgcodecs.imwrite("test3.jpg", dst);
//    	Mat roi_img = new Mat(img_gray,new Rect((int)(P[1].x+out_rect_width/14), (int)(P[1].y), out_rect_width/10*8, out_rect_height/20));
//    	
//    	Mat roi_target = roi_img.clone();
//    	
//    	double m = Core.mean(roi_target).val[0];
//    	Mat thre = new Mat(roi_target.size(),CvType.CV_8UC1,new Scalar(0));
//    	Imgproc.threshold(roi_target, thre, m - 10, m + 10, Imgproc.THRESH_BINARY);
//    	
//    	ArrayList<MatOfPoint> contours_circle = new ArrayList<MatOfPoint>();
//    	Mat hierarchy_circle = new Mat();
//    	hierarchy_circle.convertTo(hierarchy_circle, CvType.CV_8UC1);
//    	
//    	Imgproc.findContours(thre, contours_circle, hierarchy_circle, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);
//    	
//    	
//    	ArrayList<Point> circle_point = new ArrayList<Point>();
//    	for (int i = contours_circle.size()-1; i >= 0; i--) {
//    		double g_dConArea = Imgproc.contourArea(contours_circle.get(i), true);
//    		if (g_dConArea > 20 && g_dConArea < 60){
//
//    			double x_all = 0;
//    			double y_all = 0;
//    			
//    			
//    			
//    			for (int k = 0; k < contours_circle.get(i).toArray().length; k++) {
//    				x_all = x_all + contours_circle.get(i).toArray()[k].x;
//    				y_all = y_all + contours_circle.get(i).toArray()[k].y;
//    			}
//
//    			int pre_x = (int)x_all / contours_circle.get(i).toArray().length;
//    			int pre_y = (int)y_all / contours_circle.get(i).toArray().length;
//
//    			Point xy = new Point(pre_x, pre_y);
//    			circle_point.add(xy);
//    			//drawContours(roi_target, contours_circle, i, Scalar(255), 1, 8, hierarchy, hiararchyvalue + 1, Point());
//    		}
//    		
//    	}
//    	
//    	String circle_code="1";
//    	
//    	for (int i = 1 ; i < 4 ; i ++) {
//    		int side_distance = (int)(circle_point.get(i).x - circle_point.get(i - 1).x) /7 -1;	
//    		for (int k = 0 ; k < side_distance;k++) {
//    			circle_code +="0";
//    		}
//    		circle_code +="1";
//    	}
//    	code = Integer.parseInt(circle_code);

    	return 0;
    }
    
    
//  public static void doting(String address,int radius,int topMargin,int leftMargin,int dotMargin,int a[],int code_length){
//	
//	// Image Read In
//	Mat image_original = Imgcodecs.imread(address);
//	Mat image_gray = new Mat(image_original.size(),CvType.CV_8U,new Scalar(0));
//	
//	// graying 
//	Imgproc.cvtColor(image_original,image_gray, Imgproc.COLOR_BGR2GRAY);
//	
//	// Make Foreground Mask of Original Image
//	Mat mask_of_foreground = new Mat(image_gray.size(), CvType.CV_8U, new Scalar(0));
//	//Imgproc.threshold(image_gray, mask_of_foreground, 120, 255, Imgproc.THRESH_OTSU);		// 如果选择的是OTSU算法的话，阈值是多少都是无所谓的
//	Imgproc.adaptiveThreshold(image_gray, mask_of_foreground, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 3, 5);
////	
////	Mat after_ex = new Mat();
////	Mat ele = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(11,11));
////	Imgproc.erode(mask_of_foreground, after_ex, ele);
////	
////	// 提取前景。。。
////	Mat hierarchy_of_foreground = new Mat();
////    hierarchy_of_foreground.convertTo(hierarchy_of_foreground, CvType.CV_8U);
////    ArrayList<MatOfPoint> contours_foreground = new ArrayList<MatOfPoint>();
////    
////	Imgproc.findContours(after_ex, contours_foreground, hierarchy_of_foreground,Imgproc.RETR_TREE , Imgproc.CHAIN_APPROX_NONE);
////    if( contours_foreground != null && contours_foreground.size() > 0 ){
////        for(int i = 0 ; i < contours_foreground.size(); i ++){
////        	double g_dConArea = Imgproc.contourArea(contours_foreground.get(i));
////        	if (g_dConArea > 0 && g_dConArea <100000)
////    			Imgproc.drawContours(after_ex, contours_foreground, i,new Scalar(255),-1);    
////        }
////    }
////    Imgcodecs.imwrite("test.jpg", after_ex);
////	Mat mask = Mat.zeros(mask_of_foreground.rows() + 2, mask_of_foreground.cols() + 2, CvType.CV_8U);
////    Imgproc.floodFill(image_gray, mask , new Point(133,133), new Scalar(0), null, new Scalar(50), new Scalar(10), Imgproc.FLOODFILL_FIXED_RANGE);
////    
////    Imgcodecs.imwrite("test.jpg", image_gray);
//////	Mat ele = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));
//////	Imgproc.dilate(mask_foreground, mask_foreground, ele);
////    
////    Mat mask_of_foreground_reverse = new Mat(mask_of_foreground.size(),CvType.CV_8U,new Scalar(0));
////	reverse(mask_of_foreground,mask_of_foreground_reverse);
//
//	
//	
////	// 对目标背景区域进行提取
////	Mat img_canny = new Mat();
////	ArrayList<MatOfPoint> target_mask_contours = new ArrayList<MatOfPoint>();
////	
////	Imgproc.Canny(img_gray, img_canny, 30, 60);
////	
////	//Imgproc.Sobel(img_gray, img_canny, CvType.CV_8U, 5, 0);
////	    	
//////	 int scale = 1;  
//////	 int delta = 0;  
//////	 int ddepth = CvType.CV_8U;  
//////	 int kernel_size = 5;   
//////	 Imgproc.Laplacian( img_gray, img_canny, ddepth, kernel_size, scale, delta,Core.BORDER_DEFAULT );  
//////	
//////	 Imgproc.blur(img_canny, img_canny, new Size(3,3));
////	//Imgproc.morphologyEx(img_canny, img_canny, Imgproc.MORPH_CLOSE, new Mat(3,3,CvType.CV_8U));
////	
////	// 先做膨胀在做腐蚀
////	Mat ele_in = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2,2));
//////	Mat ele_out = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(2,2));
////	//Imgproc.morphologyEx(img_gray, img_canny,Imgproc.MORPH_GRADIENT ,ele_in);
////	//Imgproc.erode(img_canny, img_canny, ele_out);
////	Imgproc.dilate(img_canny,img_canny,ele_in);
////	//Imgproc.threshold(img_canny, img_canny, 10, 255, Imgproc.THRESH_BINARY);
////	
////	
////	Mat hierarchy_target_mask = new Mat();
////    hierarchy_target_mask.convertTo(hierarchy_target_mask, CvType.CV_8U);
////	Imgproc.findContours(img_canny, target_mask_contours,hierarchy_target_mask, Imgproc.RETR_EXTERNAL , Imgproc.CHAIN_APPROX_NONE);
////	
////	int max_index = 0;
////	double max_area = 0;
////	
////	for (int i = 0; i < (int)target_mask_contours.size(); i++)
////	{
////		double g_dConArea = Imgproc.contourArea(target_mask_contours.get(i));
////		//System.out.println(g_dConArea);
////		if (g_dConArea >= max_area) {
////			max_area = g_dConArea;			
////			max_index = i;
////		}
////	}
////	//System.out.println(max_index);
////	Mat target_mask = new Mat(img_canny.size(), CvType.CV_8U, new Scalar(0));
////	Imgproc.drawContours(target_mask, target_mask_contours, max_index,new Scalar(255),-1);
////	Mat new_image = new Mat(target_mask.size(),CvType.CV_8U,new Scalar(0));
////	target_mask.copyTo(new_image,reverse(mask_foreground));
////	
////	
////	ArrayList<MatOfPoint> contours_out = new ArrayList<MatOfPoint>();
////	Mat hierarchy_out = new Mat();
////    hierarchy_out.convertTo(hierarchy_out, CvType.CV_8U);
////    Imgproc.findContours(new_image, contours_out, hierarchy_out, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
////	
////    
////    MatOfPoint2f mp2f =new MatOfPoint2f( contours_out.get(0).toArray());
////    RotatedRect rect = Imgproc.minAreaRect(mp2f);
////    Point[] P = new Point[4];
////    rect.points(P);
////    
////    for(int i = 0 ; i < 4 ; i ++) {
////    	
////    	System.out.println(P[i].x + "  " + P[i].y);
////    }
////    
////    int out_rect_height = Math.abs((int)(P[0].y - P[1].y));
////	int out_rect_width  = Math.abs((int)(P[2].x - P[1].x));
////
////	Mat roi_img = new Mat(img,new Rect((int)P[1].x, (int)P[1].y, out_rect_width, out_rect_height));
////	Mat roi_fore = new Mat(mask_foreground,new Rect((int)P[1].x, (int)P[1].y, out_rect_width, out_rect_height));
////	int code_num = 0;
////	
////	Mat roi_done = new Mat(roi_img.size(), CvType.CV_8U, new Scalar(0));
////	for (int i = topMargin + radius; i - radius < roi_done.rows(); i = i + dotMargin + radius * 2){
////		for (int k = leftMargin + radius; k - radius < roi_done.cols(); k = k + dotMargin + radius * 2){
////			if (code_num <code_length) {
////				if (a[code_num] == 1) {
////					Imgproc.circle(roi_done, new Point(k, i), radius, new Scalar(255), -1, 8, 0);
////				}
////				code_num++;
////			}
////			else {
////				if (Math.random() >0.70) {
////					Imgproc.circle(roi_done, new Point(k, i), radius,new Scalar(255), -1, 8, 0);
////				}
////				else {
////					continue;
////				}
////			}
////		}
////	}
////	
////	roi_fore = reverse(roi_fore);
////	
////	
////	
////	Mat roi_test = new Mat(roi_img.size(), CvType.CV_8U, new Scalar(0));
////	Mat roi_final = new Mat(roi_img.size(), CvType.CV_8U, new Scalar(0, 0, 0));
////	roi_done.copyTo(roi_test, roi_fore);
////	
////	ArrayList<Mat> channels = new ArrayList<Mat>();
////	Core.split(roi_img, channels);
////	
//////	Mat imageGreenChannel = channels.get(1);
//////	Mat imageRedChannel   = channels.get(0);
//////	Mat imageBlueChannel  = channels.get(2);
//////
//////	//get和set方法有问题，还需要再找找
//////	for (int i = 0; i < imageGreenChannel.rows(); i++) {
//////		for (int k = 0; k < imageGreenChannel.cols(); k++) {
//////			if (roi_test.get(k,i)[0] == 255){
//////				imageGreenChannel.put(k, i,imageGreenChannel.get(k,i)[0]-35);
//////				imageRedChannel.put(k, i,imageRedChannel.get(k,i)[0]-35);
//////				imageBlueChannel.put(k, i,imageBlueChannel.get(k,i)[0]-35);
//////			}
//////		}
//////	}
//////	
//////	Core.merge(channels, roi_final);
//////	Core.addWeighted(new Mat(img,new Rect((int)P[1].x, (int)P[1].y, out_rect_width, out_rect_height)), 0.5, roi_final,
//////			0.5, 0.0, new Mat(img,new Rect((int)P[1].x, (int)P[1].y, out_rect_width, out_rect_height)));
//////	
//////	Imgcodecs.imwrite("test.jpg", img);
//}
}