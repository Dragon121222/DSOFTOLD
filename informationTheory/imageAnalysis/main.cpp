#include <iostream> 
#include <stdio.h>
#include <arrayfire.h>
#include <cstdlib>
#include <string>
#include <cstring>
#include <fstream>
#include <dirent.h>

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

static void joint_suprisal_Representation(array& input, array& output) { 

	dim_t row = input.dims(0); 
	dim_t col = input.dims(1); 

	array input_col_rep = flat(input); 

	array tmp[row*col]; 

	for(int i = 0; i < row*col; i++) { 
		tmp[i] = input_col_rep*input_col_rep(i).scalar<double>(); 
		tmp[i] = -1*tmp[i]*log(tmp[i]); 
		cout << "i: " << i << "\n"; 
	}

	for(int i = 0; i < row; i++) { 
		for(int j = 0; j < col; j++) { 
			output(i,j) = sum<double>(tmp[j + (col-1)*i]); 
		}
	}

}

static void probability_Representation(array& input, array& output) { 

	dim_t row = input.dims(0); 
	dim_t col = input.dims(1); 

	if( output.dims(0) == row && output.dims(1) == col && is_zero_vec(output) ) { 

		// Get the dimension of the collum space. 
	    dim_t input_dim = input.elements(); 

	    // Make collum representation of the input image.  
		array input_col_rep = flat(input); 

		// Get unique elements in the input image. 
		array input_unique_com = setUnique(input_col_rep,false); 

		// Find the dimension of the unique element vector. 
		dim_t unique_dim = input_unique_com.elements();

		// Make empty vector to count reoccurance of unique elements in the image.  
		array unique_count = array (unique_dim).as(f64); 

		// Set to zero vector. 
		unique_count = 0;

		array tmp1;
		array tmp2;  

		cout << "input_dim: " << input_dim << "\n"; 

		for(int i = 0; i < unique_dim; i++) { 

			if(input_unique_com(i).scalar<double>() != 0) { 
				tmp1 = ( input == input_unique_com(i).scalar<double>() ).as(f64);
				tmp2 = where(tmp1); 
				unique_count(i) = (double)tmp2.elements(); 
				tmp1 = tmp1*unique_count(i).scalar<double>()/input_dim; 
				output = output + tmp1; 
			} else { 
				cout << "Zero Value Found.\n"; 
			}

		}

		cout << "Prob Max / Min: " << max<double>(output) << " / " << min<double>(output) << "\n"; 

//		output = output/max<double>(output); 

	} else { 
		cout << "Error, missmatch.\n"; 
	}

}

static void test0(const char * fileName, bool color) {

    af::Window wnd("Information Analysis");

    array img; 
    array rr, gg, bb;
    array probRep, probRep_red, probRep_green, probRep_blue; 
    array ldif, rdif, ldif_red, rdif_red, ldif_green, rdif_green, ldif_blue, rdif_blue; 
    array absdif, absdif_red, absdif_green, absdif_blue; 
    array supprisal; 
    array joint_suprisal; 

//    array connorTransform; 

    cout << "Is color " << color << "\n"; 

    if(color) { 

    	cout << "Color Image loaded\n"; 
    	img = loadImage(fileName, true) / 255.f; // 3 channel RGB       [0-1]

	    // rgb channels

//    	connorTransform =  array(img.dims(0),img.dims(1)); 

//		connorTransform = cos(tan(123*cos(img))); 



	    channel_split(img, rr, gg, bb);

	    probRep_red = array(rr.dims(0),rr.dims(1)); 
	    probRep_red = 0;

	    probRep_green = array(gg.dims(0),gg.dims(1)); 
	    probRep_green = 0;

	    probRep_blue = array(bb.dims(0),bb.dims(1)); 
	    probRep_blue = 0; 

		probability_Representation(rr,probRep_red); 

		ldif_red   = rr - probRep_red; 
		rdif_red   = probRep_red - rr; 
		absdif_red = abs(rr - probRep_red); 

		probability_Representation(gg,probRep_green); 

		ldif_green 	 = gg - probRep_green; 
		rdif_green 	 = probRep_green - gg; 
		absdif_green = abs(gg - probRep_green); 

		probability_Representation(bb,probRep_blue); 

		ldif_blue 	= bb - probRep_blue; 
		rdif_blue 	= probRep_blue - bb; 
		absdif_blue = abs(bb - probRep_blue); 

		probRep = array(img.dims(0),img.dims(1),img.dims(2)); 
		ldif 	= array(img.dims(0),img.dims(1),img.dims(2));
		rdif 	= array(img.dims(0),img.dims(1),img.dims(2));
		absdif  = array(img.dims(0),img.dims(1),img.dims(2));

		channel_merge(probRep,probRep_red,probRep_blue,probRep_green); 
		channel_merge(ldif,ldif_red,ldif_blue,ldif_green); 
		channel_merge(rdif,rdif_red,rdif_blue,rdif_green); 
		channel_merge(absdif,absdif_red,absdif_blue,absdif_green); 		

		supprisal = -1*probRep*log(probRep); 

		supprisal = supprisal/max<float>(supprisal);

		string fn = "probRep" + string(fileName); 

//		saveImage(fn.c_str() , (probRep*255).as(f32)); 

		fn = "abs(probRep,img)" + string(fileName);

		saveImage(fn.c_str() , (absdif*255).as(f32)); 

	    while (!wnd.close()) {

	        wnd.grid(2, 1);
	        // image operations
	        wnd(0, 0).image(img, 	   "Input Image");
	        wnd(0, 1).image(probRep,   "Probability Representation");
	        wnd(1, 0).image(supprisal, "Supprisal Representation");
	        wnd(1, 1).image(1 - probRep, "Probability Representation, 1 - P");

	        wnd.show();

	    }

    } else { 

    	cout << "Greyscale Image loaded\n"; 
	    img = loadImage(fileName, false).as(f64) / 255.f;
	    probRep = array(img.dims(0),img.dims(1)); 
	    probRep = 0; 

	    img = img + 0.00000001; 

		probability_Representation(img,probRep); 

		supprisal = -1*log(probRep); 

		normalize(supprisal,supprisal); 
		normalize(probRep,probRep); 

		joint_suprisal_Representation(probRep, joint_suprisal);

		cout << "Max 1: " << max(supprisal).scalar<double>() << "\n"; 
		cout << "Max 2: " << max<double>(supprisal) << "\n"; 

//		supprisal = supprisal/max(supprisal).scalar<double>();
//		probRep = probRep/max(probRep).scalar<double>();

	    while (!wnd.close()) {

	        wnd.grid(2, 2);
	        // image operations
		        wnd(0, 0).image(img.as(f32), "Input Image, I");
		        wnd(0, 1).image(probRep.as(f32), "Probability Representation, P");
		        wnd(1, 0).image(supprisal.as(f32), "Supprisal Representation, S");
		        wnd(1, 1).image(joint_suprisal.as(f32), "Joint Suprisal Representation, J");

	        wnd.show();

	    }

    }

}

int main(int argc, char** argv) {
    int device = argc > 1 ? atoi(argv[1]) : 0;
    try {
        af::setDevice(device);
        af::info();
        printf("** Image Analysis Demo **\n\n");
        cout << "Argument count: " << argc << "\n"; 
        for(int i = 0; i < argc; i++) { 
	        cout << i << " " << argv[i] << "\n"; 
        }

    	if(argc == 3) {
    		if( strcmp(argv[2],"0")==0 ) { 
		        test0(argv[1],false);
    		} else if( strcmp(argv[2],"1")==0 ) { 
		        test0(argv[1],true);
    		} else if( strcmp(argv[2],"2")==0 ) { 

    			array * set; 
//				open_set(argv[1], set, true); 

    		} else { 

    		}
    	}    


    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
