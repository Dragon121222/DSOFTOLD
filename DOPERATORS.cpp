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

#ifndef __DOPERATORS__
	#define __DOPERATORS__
#endif

//****************************************************************************************************************
// Operators
//****************************************************************************************************************
static void matrixOperator(af::array &Input, af::array &Output, af::array &Kernel) { 
//	std::cout << "Calling Matrix Operator\n"; 
//	std::cout << "Kernel Dim: [" << Kernel.dims(0) << "," << Kernel.dims(1) << "," << Kernel.dims(2)<< "," << Kernel.dims(3) << "]\n";
//	std::cout << "Input Dim:  [" << Input.dims(0) << "," << Input.dims(1) << "," << Input.dims(2)<< "," << Input.dims(3) << "]\n"; 

	Output = af::matmul(Kernel,Input); 
}

static void additionOperator(af::array &Input, af::array &Output, af::array &Kernel) { 
//	std::cout << "Calling Addition Operator\n"; 
	Output = Input + Kernel; 
} 

static void nonLinearOperator_arcTan(af::array &Input, af::array &Output, af::array &Kernel) { 

//	std::cout << "Calling Arctan Operator\n"; 

	af::array tmp = Input; 

	af::array out = af::atan(Input)/af::Pi + 0.5; 

	Output = out; 

}

static bool is_zero_vec(af::array & input) { 
	if(af::norm(input) == 0) { 
		return true; 
	} else { 
		return false; 
	}
}

static void channel_split(af::array& rgb, af::array& outr, af::array& outg, af::array& outb) {
    outr = rgb(af::span, af::span, 0);
    outg = rgb(af::span, af::span, 1);
    outb = rgb(af::span, af::span, 2);
}

static void channel_merge(af::array& rgb_out, af::array& red, af::array& green, af::array& blue) { 
	rgb_out(af::span,af::span,0) = red; 
	rgb_out(af::span,af::span,1) = green; 
	rgb_out(af::span,af::span,2) = blue; 	
}	

static void probability_Representation(af::array& input, af::array& output) { 

	dim_t row = input.dims(0); 
	dim_t col = input.dims(1); 

//	if( output.dims(0) == row && output.dims(1) == col && is_zero_vec(output) ) { 

		// Get the dimension of the collum space. 
	    dim_t input_dim = input.elements(); 

	    // Make collum representation of the input image.  
		af::array input_col_rep = af::flat(input).as(f64); 

		// Get unique elements in the input image. 
		af::array input_unique_com = af::setUnique(input_col_rep,false).as(f64); 

		// Find the dimension of the unique element vector. 
		dim_t unique_dim = input_unique_com.elements();

		// Make empty vector to count reoccurance of unique elements in the image.  
//		af::array unique_count = af::array (unique_dim).as(f64); 
		af::array unique_count = af::array(unique_dim).as(f64); 
		// Set to zero vector. 
		unique_count = 0;

		af::array tmp1;
		af::array tmp2;  

		std::cout << "input_dim: " << input_dim << "\n"; 

		for(int i = 0; i < unique_dim; i++) { 

			if(input_unique_com(i).scalar<double>() != 0) { 
//				tmp1 = ( input == input_unique_com(i).scalar<double>() ).as(f64);
				tmp1 = ( input == input_unique_com(i).scalar<double>() ).as(f64);				
				tmp2 = af::where(tmp1); 
				unique_count(i) = (double)tmp2.elements(); 
				tmp1 = tmp1*unique_count(i).scalar<double>()/input_dim; 
				output = output + tmp1; 
			} else { 
				std::cout << "Zero Value Found.\n"; 
			}

		}

		std::cout << "Prob Max / Min: " << af::max<double>(output) << " / " << af::min<double>(output) << "\n"; 

		output = output/af::max<double>(output); 

//	} else { 
//		std::cout << "Error, missmatch.\n"; 
//	}

}

static void pairwiseDifference(std::vector<af::array> &D_Sequence) { 
	for(std::vector<af::array>::iterator it = D_Sequence.begin(); it != (D_Sequence.end() - 1); ++it) { 
		*it = *it - *(it + 1); 
	}
}

static void sequentialProbabilityRepresentation(std::vector<af::array> &D_Sequence) { 

	af::array tmp; 
	af::array rr; 
	af::array gg; 	
	af::array bb; 	

	af::array out_rr; 
	af::array out_gg; 	
	af::array out_bb;

	int i = 0; 

	for(std::vector<af::array>::iterator it = D_Sequence.begin(); it != (D_Sequence.end() - 1); ++it) { 

		*it = *it + 0.00000001;

		channel_split(*it,rr,gg,bb); 

		af::array out_rr = af::constant(0, rr.dims(0),rr.dims(1)); 
		af::array out_gg = af::constant(0, gg.dims(0),gg.dims(1));
		af::array out_bb = af::constant(0, bb.dims(0),bb.dims(1));

		probability_Representation(rr,out_rr); 
		probability_Representation(gg,out_gg); 
		probability_Representation(bb,out_bb); 		
		
		channel_merge(*it,out_rr,out_gg,out_bb); 
		std::cout << "Image: " << i << "\n"; 
		i++; 
	}
}

typedef void (*ptr2Function)(af::array &Input, af::array &Output, af::array &Kernel); 

static ptr2Function mPoint = & matrixOperator; 
static ptr2Function aPoint = & additionOperator; 
static ptr2Function nPoint = & nonLinearOperator_arcTan; 











