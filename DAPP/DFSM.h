//****************************************************************************************************************
// Written By Daniel Drake 
//****************************************************************************************************************

#include <iostream> 
#include <stdio.h>
#include <arrayfire.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <cstring>
#include <fstream>
#include <dirent.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <gtk/gtk.h>
#include <GL/glut.h>

#ifndef __DEFINES__
	#include "../defines.h"
#endif

#ifndef __DCAD__
	#include "../DCAD/DCAD.h"
#endif

#ifndef __DNN__
	#include "../DNN/DNN.h"
#endif

class DFSM { 

	public: 

		 DFSM(int argc, char **argv); 
		~DFSM();
		static void test();
		static void activate(GtkApplication *app, gpointer user_data); 


//	private: 

///	    static GtkWidget *D_window;
//		static GtkWidget *D_grid;

		static GtkWidget * TestWidget; 

}; 

