//****************************************************************************************************************
// Written By Daniel Drake 
//****************************************************************************************************************
#include <ctime>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
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
#include <gmodule.h>

#include "DNN.h"
#include "../defines.h"

#ifndef __DOPERATORS__
	#include "../DOPERATORS.cpp"
#endif


//typedef void (*ptr2Function)(af::array &Input, af::array &Output, af::array &Kernel); 

//****************************************************************************************************************
// Constructor / Deconstructors
//****************************************************************************************************************
DNN::DNN() { 

	af::setSeed(std::rand()); 

	operatorCount = 0; 

	Space_Dim_Is_Allocated = false; 

	inputSpacialDimension[0] = 0; 
	inputSpacialDimension[1] = 0; 

	minError = -1; 

}

DNN::~DNN() { 

//	std::cout << "Neural Net Deconstrctor Called\n"; 

	if(Space_Dim_Is_Allocated) { 
//		std::cout << "Space Dim Delete Called.\n"; 
		delete SpaceDim; 
	}

}

//****************************************************************************************************************
// Memory Management Controls
//****************************************************************************************************************
void DNN::allocateOperatorSpace() { 

	if(operatorCount <= 0) { 
		std::cout << "Error, Operator Count not set.\n"; 
	} else { 
		functionPointerArray = (ptr2Function*)malloc(operatorCount*sizeof(ptr2Function)); 
	}

}

void DNN::allocateKernels() { 

	std::cout << "allocateKernels called\n"; 

	if(operatorCount <= 0) { 
	
		std::cout << "Error, Operators not set.\n"; 
	
	} else { 

		int i = 0; 

		for (std::vector<std::string>::iterator it = operatorNames.begin() ; it != operatorNames.end(); ++it) { 

			#ifdef __GENERAL_DEBUG__
			    std::cout << "Operator: 		" << *it << "\n";
			    std::cout << "Input Space Dim: 	" << SpaceDim[i] << "\n";  
			    std::cout << "Output Space Dim:	" << SpaceDim[i+1] << "\n";
		    #endif

			if(*it == "Matrix Multiplication") {

				Kernels.push_back(af::randn(SpaceDim[i+1],SpaceDim[i])); 

			} else if(*it == "Vector Translation") { 

				if(SpaceDim[i] != SpaceDim[i+1]) { 
					std::cout << "Error, Input and Output spaces must match for a vector translation.\n"; 
				} else { 
					Kernels.push_back(af::randn(SpaceDim[i],1)); 
				}

			} else if(*it == "Arc-tangent") { 

				if(SpaceDim[i] != SpaceDim[i+1]) { 
					std::cout << "Error, Input and Output spaces must match for a Arc-tangent.\n"; 
				} else { 
					Kernels.push_back(af::randn(SpaceDim[i],1)); 
				}

			}

			i++; 

		}

	}

	#ifdef __GENERAL_DEBUG__
		std::cout << "allocateKernels complete\n"; 
	#endif
}

void DNN::allocateDimSpace() { 

	if(operatorCount <= 0) { 
		std::cout << "Error, Operator Count not set.\n"; 
	} else { 
		SpaceDim = (int*)malloc((operatorCount+1)*sizeof(int)); 
	}

}

//****************************************************************************************************************
// Import Data
//****************************************************************************************************************
void DNN::importTrainingInputData(std::string filePath) { 

	if(inputIsImage) {

		const char * path = filePath.c_str(); 

		#ifdef __GENERAL_DEBUG__
			std::cout << "Opening image path: " << path << "\n"; 
		#endif

		af::array tmp0 = af::loadImage( path , inputIsColor ); 

		if(inputSpacialDimension[0] == 0) { 
			inputSpacialDimension[0] = tmp0.dims(0); 
			inputSpacialDimension[1] = tmp0.dims(1); 
		}

		#ifdef __GENERAL_DEBUG__
			std::cout << "Input Space Dimension: " << inputSpacialDimension[0] << "*" << inputSpacialDimension[1]; 
			std::cout << " = " << inputSpacialDimension[0]*inputSpacialDimension[1] << "\n"; 
		#endif

		af::array tmp1 = flat(tmp0);

		trainingImages.push_back(tmp1); 

	}

}

void DNN::importTrainingLabelData(std::string filePath) { 

	if(outputIsTextIntegers) {

		const char * path = filePath.c_str(); 
		std::string line; 
		std::ifstream file;
		int val[SpaceDim[operatorCount]]; 

		int i = 0; 

		std::cout << "Label imported: \n"; 

		file.open(path); 
		if (file.is_open()) {
			while ( getline (file,line) ) {
				val[i] = std::stoi(line); 
				std::cout << val[i] << "\n";
				i++; 
			}
			file.close();
		} else { 
			std::cout << "File would not open\n"; 
			std::cout << "Path: " << filePath << "\n"; 
		}

		af::array tmp = af::array(SpaceDim[operatorCount],1,val); 

		trainingLabels.push_back(tmp); 

	} else { 
		std::cout << "Error, output handeling not programmed yet.\n"; 
	}

}

//****************************************************************************************************************
// Generate Data
//****************************************************************************************************************
void DNN::generateRandomNonRepeatingIntegerSequence(int count) { 

	imagePerCategoryCount = count; 

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	for(int i = 1; i <= imagePerCategoryCount; i++) { 
		randomIndicies.push_back(i); 
	}

	shuffle (randomIndicies.begin(), randomIndicies.end(), std::default_random_engine(seed));

	#ifdef __GENERAL_DEBUG__
		for (std::vector<int>::iterator it = randomIndicies.begin() ; it != randomIndicies.end(); ++it) { 
			std::cout << *it << " "; 
		}
		std::cout << "\n";
	#endif

}

int getPreviousReportNumber() { 

	std::ifstream file; 	
	std::string line; 
	int num = -1; 
	file.open("/home/drake/Documents/Code/DSOFT/AppData/expNum"); 
	if (file.is_open()) {
		getline (file,line);
		num = std::stoi( line ); 
		file.close();
	} else { 
		std::cout << "File would not open\n"; 
	}
	return num; 

}

void setReportNumber(int num) { 

    std::ofstream file("/home/drake/Documents/Code/DSOFT/AppData/expNum");
    if(file.is_open()) {
        file << num;             
    } else {
		std::cout << "File would not open\n"; 
    }

}

void DNN::generateDocumentation() { 

	std::string document = ""; 

   // current date/time based on current system
   time_t now = time(0);
   
   // convert now to string form
   std::string current_Time = std::string(ctime(&now));


	#ifdef __DOC_DEBUG__

   		int reportNumber = getPreviousReportNumber()+1; 

   		std::cout << "\n\nFinite Composition Operator Experiment Report: " << reportNumber << "\n"; 
		setReportNumber(reportNumber); 
		std::cout << "Date/Time: " << current_Time << "\n"; 

		std::cout << "Training Set Selected: " << trainingSetName << "\n";
		std::cout << "	Number of Categories: " << categoryCount << "\n"; 
		std::cout << "	Number of input vectors per category: " << imagePerCategoryCount << "\n"; 
		std::cout << "	Total Number of input vectors: " << imagePerCategoryCount*categoryCount << "\n\n"; 

		std::cout << "Training Algorithm Selected: " << trainingAlgorthmName << "\n"; 
		std::cout << "	Trial Count: " << trialCount << "\n"; 
		std::cout << "	Min Error Value: " << minError << "\n\n"; 

		std::cout << "Preprocessing Algorithm Selected: " << preprocessingAlgorthmName << "\n\n"; 

		std::cout << "Operators Selected: " << "\n"; 

		int i = 0; 

		for(std::vector<std::string>::iterator it = operatorNames.begin(); it != operatorNames.end(); ++it) { 
			std::cout << *it << "\n"; 
			std::cout << "Mapping: " << SpaceDim[i] << "->" << SpaceDim[i+1] << "\n\n"; 
			i++; 
		}

	#endif

}

//****************************************************************************************************************
// Set Controls		
//****************************************************************************************************************
void DNN::setOperatorCount(int count) { 

	if(count > 0) { 
		operatorCount = count; 
	} else { 
		std::cout << "Error, only defined for positive integers.\n"; 
	}

}

void DNN::setSpaceDim(int index, int dim) { 
	if(0 <= index && index <= operatorCount && 0 < dim) { 
		SpaceDim[index] = dim; 
	} else { 
		std::cout << "Error in set Space Dim Function\n"; 
	}
}

void DNN::setOperator(int index,void(functionPointer)(af::array &Input,af::array &Output,af::array &Kernel), std::string operatorName) { 

	functionPointerArray[index] = functionPointer; 

	operatorNames.push_back(operatorName); 

}

void DNN::setCategoryCount(int count) { 
	categoryCount = count; 
}

void DNN::setImagePerCategoryCount(int count) { 
	imagePerCategoryCount = count; 
}

void DNN::setInformationAboutInputSpace(bool isImage, bool isColor) { 
	inputIsImage = isImage; 
	inputIsColor = isColor; 	
}

void DNN::setInformationAboutOutputSpace(bool isTextIntegers) { 
	outputIsTextIntegers = isTextIntegers; 
}

void DNN::setInformationAboutTrainingSpace(std::string name) { 
	trainingSetName = name; 
}

void DNN::setInformationAboutTrainingAlgorithm(std::string name) { 
	trainingAlgorthmName = name; 
}

void DNN::setInformationAboutPreprocessingAlgorithm(std::string name) { 
	preprocessingAlgorthmName = name; 
}


//****************************************************************************************************************
// Get Controls
//****************************************************************************************************************
int DNN::getOperatorCount() { 
	return operatorCount; 
}

//****************************************************************************************************************
// Operational Functions
//****************************************************************************************************************
void DNN::fire(af::array &Input,af::array &Output) { 

	af::array temp0 = Input; 
	af::array temp1; 

	int i = 0; 

	for (std::vector<af::array>::iterator it = Kernels.begin() ; it != Kernels.end(); ++it) { 

		functionPointerArray[i](temp0,temp1,*it); 
		temp0 = temp1; 
		i++; 

	}

	Output = temp0; 

}

//****************************************************************************************************************
// Preprocessing 
//****************************************************************************************************************
void DNN::remove_Outliers() { 

}

//****************************************************************************************************************
// Training Algorthms
//****************************************************************************************************************
void DNN::gradient_Descent() { 
}

void DNN::sub_Gradient_Method() { 

}

void DNN::random_Trials() { 

	std::cout << "Start Random Trials\n"; 

	std::cout << "Enter Number of Trials\n"; 

	std::cin >> trialCount; 

	for(int i = 0; i < trialCount; i++) { 
		std::cout << "Trial: " << i << "\n"; 
		double error = test_Data_Set();  
		store_Kernel(error); 
		randomize_Kernel(); 

	}

	std::cout << "Random Trials are Complete\n"; 	

}

double DNN::test_Data_Set() { 

	std::vector<af::array>::iterator iit = trainingImages.begin(); 
	std::vector<af::array>::iterator oit = trainingLabels.begin(); 

	af::array tmp; 

	double error = 0; 

	iit++; 
	oit++; 

	while( iit != trainingImages.end() ) { 

		fire(*iit,tmp); 

		error += norm( (*oit) - tmp); 

		iit++; 
		oit++; 

		if(error > minError && minError != -1) { 
			break;
		}

	}

	std::cout << "Error: " << error << "\n"; 
	return error; 

}

void DNN::store_Kernel(double e) { 
	
	if(e < minError) { 
		minError = e; 
	} else if(minError == -1) { 
		minError = e; 
	}

}

void DNN::randomize_Kernel() { 

	std::cout << "Randomizing Kernel\n"; 
	int i = 0; 
	for(std::vector<af::array>::iterator it = Kernels.begin() ; it != Kernels.end(); ++it) { 
		*it = af::randn((*it).dims(0),(*it).dims(1)); 
		i++; 
	}

}

//****************************************************************************************************************
// Debug Functions
//****************************************************************************************************************
void DNN::debugKernels() { 

	for (std::vector<af::array>::iterator it = Kernels.begin() ; it != Kernels.end(); ++it) { 
		#ifdef __GENERAL_DEBUG__
		    af::print("",*it); 
		    std::cout << "\n"; 
	    #endif
	}

}

void DNN::inspectTrainingSet() { 

	if(inputIsImage && !inputIsColor && outputIsTextIntegers) {

		std::cout << "Inspecting Training Sets\n"; 

	    af::Window afWindow("Training Set Inspector");

	    std::vector<af::array>::iterator iit = trainingImages.begin(); 
	    std::vector<af::array>::iterator oit = trainingLabels.begin(); 

		af::array tmp = wrap(	*iit, 
								inputSpacialDimension[0], inputSpacialDimension[1], 
								inputSpacialDimension[0], inputSpacialDimension[1],
								1,1,0,0,
								true
								); 

		std::string input = " "; 

	    std::cout << "Enter +c to move forward to the next category\n"; 
	    std::cout << "Enter ++ to move forward to the next image\n"; 
	    std::cout << "Enter -c to move back to the previous category\n"; 
	    std::cout << "Enter -- to move back to the previous image\n";     
	    std::cout << "Enter q close.\n";     

	    while(!afWindow.close() && input != "q") { 
	        afWindow.image(tmp);

	        af::print("Output Training Set: ",*oit); 

	        std::cout << "Enter: "; 
	        std::cin >> input; 

	        if(input == "+c") { 
	        	iit += imagePerCategoryCount; 
	        	oit += imagePerCategoryCount; 	        	
	        } else if(input == "++") { 
	        	iit++;
	        	oit++; 
	        } else if(input == "-c") { 
	        	iit -= imagePerCategoryCount; 
	        	oit -= imagePerCategoryCount; 	        	
	        } else if(input == "--") { 
	        	iit--;
	        	oit--; 
	        } else if(input == "q") { 

	        } else { 
	        	std::cout << "Input: " << input << " is not valid\n"; 
	        }

			tmp = wrap(	*iit, 
						inputSpacialDimension[0], inputSpacialDimension[1], 
						inputSpacialDimension[0], inputSpacialDimension[1],
						1,1,0,0,
						true
						); 

	    }

	}

}
