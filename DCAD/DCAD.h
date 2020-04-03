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
	#include "DMESH.h"
#endif

#ifndef __DCAD__
	#define __DCAD__
#endif

#ifndef __DEFINES__
	#include "../defines.h"
#endif

class DCAD { 

	public: 

		 DCAD(); 
		~DCAD(); 

	private: 

		// Hard Coded Stuff
		DMESH unitCube; 
		std::string directory = "models/";
		std::string unitCubeFileName = directory + "unitCube.stl"; 


		// Procedural Stuff
		DMESH ** meshHierarchy; 

}; 