#include <iostream> 
#include <stdio.h>
#include <arrayfire.h>
#include <cstdlib>
#include <string>
#include <cstring>
#include <fstream>
#include <dirent.h>
#include <stdlib.h>
#include <time.h>

#include "DSUDOKUGEN.h"
#include "../defines.h"

using namespace af;
using namespace std; 

sudokuGenerator::sudokuGenerator() { 
	srand (time(NULL));
	values = array(9,9); 
} 

sudokuGenerator::~sudokuGenerator() { 

}

void sudokuGenerator::generate() { 

	bool namesAreHard = false; 
	bool accepted = false; 
	float sum = 0; 
	float val; 


	for(int j = 0; j < 9; j++) { 

		while(!accepted) { 
			
			accepted = true; 

			val = rand() % 9 + 1; 

			for(int i = 0; i < j; i++) { 
				if(val == values(0,i).scalar<float>() ) { 
					accepted = false; 
				}
			}
		}

		if(accepted) { 
			values(0,j) = val; 
			accepted = false; 
		}

	}


	for(int j = 0; j < 9; j++) { 
		cout << values(0,j).scalar<float>() << "+";  
	}

}


