#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream> 
#include <stdio.h>
#include <arrayfire.h>
#include <GL/glut.h>
#include <cstdlib>
#include <string>
#include <cstring>
#include <fstream>
#include <dirent.h>

#ifndef __DMESH__
	#define __DMESH__
#endif

#ifndef __DEFINES__
	#include "../defines.h"
#endif

//using namespace af;
//using namespace std; 

class DMESH { 

	public: 

		// Constructors

		 DMESH(); 
		~DMESH(); 

		// Memory Management

		void allocateTriangleMemory(int triangleCount); 
		void deAllocateTriangleMemory(); 

		// External Functions

		void drawMesh(); 
		void importMesh(std::string &fileName); 
		void exportMesh(std::string &fileName); 

		// Maker Functions

		void makeCuboid(); 

		// Modifier Functions

		void translate(af::array &translationVector); 
		void scale(double &scalar); 
		void rotate(af::array &rotationMatrix); 

	private: 

		af::array triangleList; 
		int triCount; 
		bool set; 

};