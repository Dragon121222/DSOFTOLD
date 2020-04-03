#include <iostream> 
#include <stdio.h>
#include <arrayfire.h>
#include <cstdlib>
#include <string>
#include <cstring>
#include <fstream>
#include <dirent.h>
#include "neuralNet.h"
#include "sudokuGenerator.h"
#include "defines.h"

using namespace af;
using namespace std; 

// Split a MxNx3 image into 3 separate channel matrices.
static void channel_split(array& rgb, array& outr, array& outg, array& outb) {
    outr = rgb(span, span, 0);
    outg = rgb(span, span, 1);
    outb = rgb(span, span, 2);
}

static void channel_merge(array& rgb_out, array& red, array& green, array& blue) { 
	rgb_out(span,span,0) = red; 
	rgb_out(span,span,1) = green; 
	rgb_out(span,span,2) = blue; 	
}	

static bool is_zero_vec(array & input) { 
	if(norm(input) == 0) { 
		return true; 
	} else { 
		return false; 
	}
}

static void normalize(array& out, array& in) { 
	out = in - min<float>(in); 
	out = out / max<float>(out);  
}	

void test0() { 


	    neuralNet net = neuralNet(); 
/*
	    array x = array(2); 
	    x(0) = 1; 
	    x(1) = 0;         
	    array y = array(2); 

	    net.fire(x,y); 

	    array d_1 = array(2); 

		array v = randn(2,1,1); 

		v = v/norm(v); 

		net.compute_Quasisecant(x, d_1, v); 
*/

		array a = randn(2,1,1); 
		array b = randn(2,1,1); 

		print("a: ", a); 
		print("b: ", b); 		

		net.minimize_convex(a, b); 


//	    double error = net.calculateError(x,y); 

//	    cout << "Error: " << error << "\n"; 


//	sudokuGenerator gen = sudokuGenerator(); 

//	gen.generate(); 

}

int main(int argc, char** argv) {

	srand (time(NULL));

    int device = argc > 1 ? atoi(argv[1]) : 0;
    try {
        af::setDevice(device);
        af::info();
        printf("** Image Analysis Demo **\n\n");

        test0(); 


    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
