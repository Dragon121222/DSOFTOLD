//****************************************************************************************************************
// Written By Daniel Drake 
//****************************************************************************************************************

#include <vector>
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

//using namespace af;
//using namespace std; 

class DNN { 

	public: 

		DNN(); 
		~DNN(); 

		// Memory Management Controls
		void allocateOperatorSpace(); 
		void allocateKernels(); 
		void allocateDimSpace(); 

		// Import Data
		void importTrainingInputData(std::string filePath); 
		void importTrainingLabelData(std::string filePath); 

		// Generate Data
		void generateRandomNonRepeatingIntegerSequence(int count); 
		void generateDocumentation(); 

		// Set Controls		
		void setOperatorCount(int count); 
		void setSpaceDim(int index, int dim); 
		void setOperator( int index, void (functionPointer)(af::array &Input, af::array &Output, af::array &Kernel), std::string operatorName);
		void setCategoryCount(int count); 
		void setImagePerCategoryCount(int count); 
		void setInformationAboutInputSpace(bool isImage, bool isColor ); 
		void setInformationAboutOutputSpace(bool isTextIntegers); 
		void setInformationAboutTrainingSpace(std::string name); 
		void setInformationAboutTrainingAlgorithm(std::string name); 
		void setInformationAboutPreprocessingAlgorithm(std::string name); 

		// Get Controls
		int getOperatorCount(); 

		// Operational Functions
		void fire(af::array &Input,af::array &Output); 
		double test_Data_Set(); 
		void store_Kernel(double e); 
		void randomize_Kernel(); 

		// Preprocessing 
		void remove_Outliers();

		// Training Algorthms
		void gradient_Descent();
		void sub_Gradient_Method();
		void random_Trials(); 

		// Debug Functions
		void debugKernels(); 
		void inspectTrainingSet(); 

	private: 

		std::vector<int>		randomIndicies; 
		int 					categoryCount; 
		int 					imagePerCategoryCount; 

		std::string 			trainingSetName; 
		std::string 			preprocessingAlgorthmName; 
		std::string 			trainingAlgorthmName;

		std::vector<af::array>	trainingImages; 
		int 					inputSpacialDimension[2]; 
		bool 					inputIsImage; 
		bool 					inputIsColor; 
		af::timer 				inputDataImportTimer; 

		std::vector<af::array>	trainingLabels; 
		bool 					outputIsTextIntegers; 
		af::timer 				lableImportTimer; 

		int 					trialCount; 
		int 					minError; 

		int 					operatorCount; 

		int * 					SpaceDim; 
		bool 					Space_Dim_Is_Allocated; 

		void (**functionPointerArray)(af::array &Input, af::array &Output, af::array &Kernel) = NULL;
		af::timer operatorTimer; 
		double operatorTimerAverage; 

		std::vector<af::array> Kernels; 

		std::vector<std::string> operatorNames; 

//		bool Kernel_Is_Allocated; 

};


