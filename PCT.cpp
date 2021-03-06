#include <iostream>
#include <string>  
#include <iomanip> 
#include <sstream> 
#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>        
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv/cv.h"
#include "PCT.h"


using namespace xfeatures2d;

void PCTable::init()
{

	table.resize(500);
	entry_num = 0;

}


int PCTable::tableSize() 
{
	return entry_num;
}

void PCTable::addAllEntries(vector<KeyPoint> twoDee, vector<SpacePoint> threeDee) 
{
	int size2d = static_cast<int>(twoDee.size());
	int threeD_Start = static_cast<int>(threeDee.size() - size2d);

	cout << "size2d is " << size2d << ", 3d size is " << threeDee.size() << endl;


	for (int i = 0; i < size2d; i++) 
	{
		Point2d *twoD = (Point2d *) malloc(sizeof(Point2d));
		Point3d *threeD = (Point3d *) malloc(sizeof(Point3d));

		twoD->x = twoDee.at(i).pt.x;
		twoD->y = twoDee.at(i).pt.y;

		threeD->x = threeDee.at(threeD_Start + i).point.x;
		threeD->y = threeDee.at(threeD_Start + i).point.y;
		threeD->z = threeDee.at(threeD_Start + i).point.z;

		add_entry(threeD, twoD);
	}

	cout << "Writing to file..." << endl;
	ofstream outfile ("table.txt");
	for (int i = 0; i < entry_num - 1; i++) {
		outfile << i << ") 3d:" << table.at(i).cp->x << " " << table.at(i).cp->y << " " << table.at(i).cp->z << "      2d: " << table.at(i).fp->x << " " << table.at(i).fp->y << "\n";
	}
	outfile.close();

}

void PCTable::add_entry(Point3d *cp, Point2d *fp) 
{

	if(entry_num >= table.size())
	{
		table.resize(500+table.size());
	}

	Entry e;
	e.fp = fp;
	e.cp = cp;
	table[entry_num] = e;
	entry_num++;

	
}


PCTable::PCTable() 
{
}

Point3d* PCTable::find_3d(Point2d fp)
{


	int i;
	Point3d *p =NULL;

	for(i = 0; i < entry_num; i++)
	{

		if(table[i].fp->x == fp.x && table[i].fp->y == fp.y)
		{
			p = table[i].cp;
			return p;
		}
	}

	return p;
}	

void PCTable::cleanup(){
	for (int i = 1; i < entry_num - 1; i++) 
	{
		if (table[i].fp != NULL)
		{
			table[i].fp = NULL;
		}
		if (table[i].cp != NULL)
		{
			free(table[i].cp);
			table[i].cp = NULL;
		}
		
	}

	table.resize(0);
	entry_num = 0;
}


bool PCTable::test( )
{
	int i =0,k = 1;
	PCTable l;
	l.init();
	while(k < 30000){

		Point2d *a = (Point2d *)malloc(sizeof(Point2d));
		Point3d *b = (Point3d *)malloc(sizeof(Point3d));

		a->x = 4 + i;
		a->y = 6 + i;

		b->x = 1 + i;
		b->y = 2 + i;
		b->z = 5 + i;

		i++;
		k++;

		l.add_entry(b, a);
	}

	Point2d m;
	m.x = 1000;
	m.y = 1002;

	Point3d *c = l.find_3d(m);

	cout << c->x << " " << c->y << " "<< c->z << endl;
	l.cleanup();
	if(c->x == 997 && c->y == 998 && c->z == 1001) 
	{
		return true;

	} else 
	{
		return false;
	}
		return true;
}
