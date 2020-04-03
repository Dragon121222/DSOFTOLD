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

class sudokuGenerator { 

	public: 

		sudokuGenerator(); 
		~sudokuGenerator(); 
		void generate(); 

		array values; 

};
