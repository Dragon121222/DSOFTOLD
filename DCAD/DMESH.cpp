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

#ifndef __DEFINES__
	#include "../defines.h"
#endif

//using namespace af;
//using namespace std; 

// Constructor
DMESH::DMESH() { 

} 

// Deconstructor
DMESH::~DMESH() { 

}

// Memory Management

// Triangle 1
// [x0,y0,z0]
// [x1,y1,z1]
// [x2,y2,z2]
// Triangle 2
// [x3,y3,z3]
// [x4,y4,z4]
// [x5,y5,z5]
// .
// .
// .
// Triangle n
// [x3n-3,y3n-3,z3n-3]
// [x3n-2,y3n-2,z3n-2]
// [x3n-1,y3n-1,z3n-1]

void DMESH::allocateTriangleMemory(int triangleCount) { 
	triCount = triangleCount; 
	triangleList = af::array(3*triCount,3).as(f64);
}

void DMESH::deAllocateTriangleMemory() { 

}

void DMESH::drawMesh() { 

	glBegin(GL_TRIANGLES); 

		for(int i = 0; i < triCount; i++) { 
	
	    glColor3f(((double)i)/triCount,((double)i)/triCount,((double)i)/triCount);

			glVertex3d(	triangleList(3*i,0).scalar<double>(),
						triangleList(3*i,1).scalar<double>(),	
						triangleList(3*i,2).scalar<double>() 
						);

			glVertex3d(	triangleList(3*i+1,0).scalar<double>(),
						triangleList(3*i+1,1).scalar<double>(),	
						triangleList(3*i+1,2).scalar<double>() 
						);

			glVertex3d(	triangleList(3*i+2,0).scalar<double>(),
						triangleList(3*i+2,1).scalar<double>(),	
						triangleList(3*i+2,2).scalar<double>() 
						);					

		}

	glEnd();

	glBegin(GL_LINES); 
    glColor3f(0.1f,0.1f,0.1f);
    glLineWidth(2); 

		for(int i = 0; i < triCount; i++) { 

			glVertex3d(	triangleList(3*i,0).scalar<double>(),
						triangleList(3*i,1).scalar<double>(),	
						triangleList(3*i,2).scalar<double>() 
						);
											
			glVertex3d(	triangleList(3*i+1,0).scalar<double>(),
						triangleList(3*i+1,1).scalar<double>(),	
						triangleList(3*i+1,2).scalar<double>() 
						);

			glVertex3d(	triangleList(3*i+2,0).scalar<double>(),
						triangleList(3*i+2,1).scalar<double>(),	
						triangleList(3*i+2,2).scalar<double>() 
						);					
			glVertex3d(	triangleList(3*i,0).scalar<double>(),
						triangleList(3*i,1).scalar<double>(),	
						triangleList(3*i,2).scalar<double>() 
						);

		}

	glEnd(); 

} 

void DMESH::importMesh(std::string &fileName) { 

	std::cout << "Opening File: " << fileName << "\n"; 
	boost::regex expr{"(-\\d\\.\\d{6}|\\d\\.\\d{6})"};
	boost::smatch what;
	std::fstream file; 
	std::string line; 
	file.open(fileName); 
	double value = 0; 
	int counter = 0; 
	int vertexCount = 0; 
	int componentCounter = 0;

	if(file.is_open()) {
		while(getline(file,line)) {

			boost::regex_token_iterator<std::string::iterator> it{line.begin(), line.end(), expr, 1};
			boost::regex_token_iterator<std::string::iterator> end;

			std::cout << counter << " : " << line << "\n"; 

			while (it != end) { 

			    value = boost::lexical_cast<double>(*it++);

				if( ((counter-1) % 14 == 0) || ((counter - 8) % 14 == 0)) { 
					// Do something with the normal components
				} else { 

					std::cout << value << " "; 
					triangleList(vertexCount,componentCounter) = value; 
					componentCounter++; 
					if(componentCounter == 3) { 
						componentCounter = 0; 
						vertexCount++; 
					}

				}

			}

			std::cout << "\n\n";					
			counter++;

		}
		file.close();
	}

	af_print(triangleList); 

} 

void DMESH::exportMesh(std::string &fileName) { 

} 

// Maker Functions
void DMESH::makeCuboid() { 

}

// Modifier Functions
void DMESH::translate(af::array &translationVector) { 

}

void DMESH::scale(double &scalar) { 
	triangleList = triangleList*scalar; 
}

void DMESH::rotate(af::array &rotationMatrix) { 

}