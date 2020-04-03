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
#include <gtk/gtk.h>
#include <GL/glut.h>
#include <gmodule.h>
#include <sys/socket.h> 
#include <netinet/in.h> 

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

//#include <gtkhtml/gtkhtml.h>
//#include <gtkhtml/gtkhtml-stream.h>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

namespace fs = boost::filesystem;

#ifndef __DFSM__
	#include "DFSM.h"
#endif

#ifndef __DCAD__
	#include "../DCAD/DCAD.h"
#endif

#ifndef __DMESH__
	#include "DMESH.h"
#endif

#ifndef __DEFINES__
	#include "../defines.h"
#endif

#ifndef __DOPERATORS__
	#include "../DOPERATORS.cpp"
#endif

static void print_hello(GtkWidget *widget, gpointer data); 
static void open_camera(GtkWidget *widget, gpointer data); 
static void _3dcad(GtkWidget *widget, gpointer data);
static void create_Active_Neural_Net(GtkWidget *widget, gpointer *data);
static void construct_Neural_Net(GtkWidget *widget, gpointer data);
static void activate(GtkApplication *app, gpointer user_data);
static void Open_Heiarchey(GtkWidget *widget, gpointer *data);
static void New_Heiarchey(GtkWidget *widget, gpointer *data);
static void HardCoded_Heiarchey(GtkWidget *widget, gpointer *data);
static void open_image(GtkWidget *widget, gpointer *data); 
static void open_website(GtkWidget *widget, gpointer *data); 
static void display_Html(GtkWidget *widget, gpointer *data); 
static void select_Training_Set(); 
static void start_Training(GtkWidget *widget, gpointer data);
static void create_experiment_report(GtkWidget *widget, gpointer *data); 
static void DVideoToPictures(GtkWidget *widget, gpointer data); 
static void DVideoProcessing(GtkWidget *widget, gpointer data); 
static void DApplyTransform(GtkWidget *widget, gpointer &data); 
static void DViewImages(GtkWidget *widget, gpointer data); 
static void ping(GtkWidget *widget, gpointer data); 
static void D_Client_Server_Pair(GtkWidget *widget, gpointer data); 
static void runServer(GtkWidget *widget, gpointer data); 
static void runClient(GtkWidget *widget, gpointer data); 
static void D_record_Video(GtkWidget *widget, gpointer path); 
static void D_Save_Images(GtkWidget *widget, gpointer data); 
static void D_Open_Audacity(GtkWidget *widget, gpointer data); 

// Operators
static void matrixOperator(af::array &Input, af::array &Output, af::array &Kernel);
static void additionOperator(af::array &Input, af::array &Output, af::array &Kernel);
static void nonLinearOperator_arcTan(af::array &Input, af::array &Output, af::array &Kernel);
static void import_Video_Picture_List(GtkWidget *widget, gpointer &data); 

std::string slistFilesInDirectory(std::string pathToDirectory);
void vlistFilesInDirectory(std::string pathToDirectory, std::vector<std::string> &output); 
void ShowMessage(const char* msg, const char* title); 
void FileChooser(GtkWindow* window, void (*open_file_function)(char * filename) );
void FileSelector(GtkWindow* window, std::string &fileSelected );  
void default_open_file_function(char * filename); 
void read_Combo(GtkWidget * combo); 
void display_image(char * filename); 
void read_Field(GtkWidget *widget, gpointer *data); 
void add_View(GtkWidget * D_View, GtkTextBuffer * D_Buffer, GtkWidget * D_Grid, char * D_Text, int row, int col); 
void get_Html_From_Website(GtkEntryBuffer * url_buffer, GtkEntryBuffer *output_buffer); 
std::string getFileNameFromPath(std::string path); 
std::string getLastWordAfterDelimiter(std::string path, char d_lim); 
std::string getWordsBeforeDelimiter(std::string path, char d_lim); 
void write_text_buffer_vectors_to_memory(std::vector<GtkTextBuffer*> input, std::string path ); 
void D_save_openCV_Video(std::vector<cv::Mat> video);
void D_play_recorded_video(std::vector<cv::Mat> video); 
void import_Minst_Dataset(int numberPerCategory); 
void D_update_time(); 

int status;

// GTK Global App
GtkApplication *app;

DCAD * graphics_Heiarchey; 

DNN * Neural_Net; 

static std::string current_Time; 

int opCount; 

static GtkWidget *D_Report_Topic; 
static GtkWidget *D_Report_hypothesis; 
static GtkWidget *D_Report_Procedure; 
static GtkWidget *D_Report_Results;
static GtkWidget *D_Report_Analysis;

static std::string D_videoPath = "/home/drake/Videos/Rep_"; 
static std::string D_audioPath = "/home/drake/Audio/Rep_"; 
static std::string D_ImagePath = "/home/drake/Pictures/Rep_"; 
static std::string D_ImageListPath = "/home/drake/Lists/Rep_"; 

static GtkWidget *D_Video_Select_Combo; 
static GtkWidget *D_Video_Transform_Select_Combo; 
static bool open_File_Trigger; 
static bool apply_Transform_Trigger;
static std::vector<af::array> images; 

static GtkEntryBuffer *D_Target_IP_Buffer; 	

static GtkEntryBuffer * D_Website_Select_Field_Buffer;  

static GtkEntryBuffer * D_Operator_Count_Buffer;  
static GtkEntryBuffer ** D_Operator_Name_Buffers;  
static GtkEntryBuffer ** D_Operator_Dimension_Buffers;  
static GtkWidget ** D_Operator_Select_Combo;  

static GtkWidget *** D_Buffer_Collection; 

static GtkWidget **D_Training_Setup_Data; 

//****************************************************************************************************************
// Static Member Declerations
//****************************************************************************************************************

GtkWidget *DFSM::TestWidget = NULL; 

//****************************************************************************************************************
// Constructor
//****************************************************************************************************************

DFSM::DFSM(int argc, char **argv) { 

	Neural_Net = new DNN(); 

	D_Operator_Count_Buffer = gtk_entry_buffer_new (NULL,-1);
	D_Website_Select_Field_Buffer = gtk_entry_buffer_new (NULL,-1);
	D_Target_IP_Buffer = gtk_entry_buffer_new (NULL,-1);

    app = gtk_application_new("org.gtk.example", G_APPLICATION_FLAGS_NONE);
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    status = g_application_run(G_APPLICATION(app), argc, argv);

	open_File_Trigger = false; 
	apply_Transform_Trigger = false; 



}

//****************************************************************************************************************
// Deconstructor
//****************************************************************************************************************

DFSM::~DFSM() { 

    if(opCount > 0) { 

    	delete D_Operator_Name_Buffers;
    	delete D_Operator_Dimension_Buffers; 
    	delete D_Operator_Select_Combo; 
    	delete D_Buffer_Collection; 
    	delete D_Training_Setup_Data; 
    }

    Neural_Net->~DNN(); 

    g_object_unref(app);

    std::cout << "Deconstructor Called\n"; 

}

void DFSM::test() { 
	std::cout << "Hello\n"; 
}

//****************************************************************************************************************
// Chat Client / Server
//****************************************************************************************************************

static void D_Client_Server_Pair(GtkWidget *widget, gpointer data) { 

	std::cout << "Client Server Pair\n"; 

	// Setup stuff
	GtkWidget *D_window;
	GtkWidget *D_grid;

	GtkWidget *D_Target_IP_Label;
	GtkWidget *D_Target_IP_Field; 	

	GtkWidget *D_Ping_Button;
	GtkWidget *D_runServer_Button;
	GtkWidget *D_runClient_Button;		 

	//****************************************************************************************************************
	// Default Setup
	//****************************************************************************************************************

	// Window Setup
    D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "Client Server Pair");

    // Grid Setup
	D_grid = gtk_grid_new ();
	gtk_container_add (GTK_CONTAINER (D_window), D_grid);

	D_Target_IP_Label = gtk_label_new ("Enter Target IP: ");
	gtk_grid_attach (GTK_GRID (D_grid), D_Target_IP_Label, 0, 0, 1, 1);	

	D_Target_IP_Field = gtk_entry_new_with_buffer (D_Target_IP_Buffer);
	gtk_grid_attach (GTK_GRID (D_grid), D_Target_IP_Field, 1, 0, 1, 1);	

	D_Ping_Button = gtk_button_new_with_label ("Ping");
	g_signal_connect (D_Ping_Button, "clicked", G_CALLBACK (ping), NULL);
	g_signal_connect_swapped (D_Ping_Button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_Ping_Button, 0, 2, 1, 1);

	D_runServer_Button = gtk_button_new_with_label ("Run Server");
	g_signal_connect (D_runServer_Button, "clicked", G_CALLBACK (runServer), NULL);
	g_signal_connect_swapped (D_runServer_Button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_runServer_Button, 0, 3, 1, 1);

	D_runClient_Button = gtk_button_new_with_label ("Run Client");
	g_signal_connect (D_runClient_Button, "clicked", G_CALLBACK (runClient), NULL);
	g_signal_connect_swapped (D_runClient_Button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_runClient_Button, 0, 4, 1, 1);		

	gtk_widget_show_all (D_window);
}

static void ping(GtkWidget *widget, gpointer data) {
    const char * val = gtk_entry_buffer_get_text( (GtkEntryBuffer *)  D_Target_IP_Buffer);	
	system( (std::string("ping -c 10 ") + std::string(val)).c_str() ); 
	D_Client_Server_Pair(NULL,NULL); 
} 

static void runServer(GtkWidget *widget, gpointer data) { 

    int server_fd, new_socket, valread; 
    struct sockaddr_in address; 
    int opt = 1; 
    int addrlen = sizeof(address); 
    char buffer[1024] = {0}; 
    char *hello = (char*)std::string("Hello from server").c_str(); 
       
    // Creating socket file descriptor 
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) { 
        perror("socket failed"); 
        exit(EXIT_FAILURE); 
    } 
       
    // Forcefully attaching socket to the port 8080 
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) { 
        perror("setsockopt"); 
        exit(EXIT_FAILURE); 
    } 
    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = INADDR_ANY; 
    address.sin_port = htons( PORT ); 
       
    // Forcefully attaching socket to the port 8080 
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address))<0) { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 
    if (listen(server_fd, 3) < 0) 
    { 
        perror("listen"); 
        exit(EXIT_FAILURE); 
    } 
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen))<0) { 
        perror("accept"); 
        exit(EXIT_FAILURE); 
    } 
    valread = read( new_socket , buffer, 1024); 
    printf("%s\n",buffer ); 
    send(new_socket , hello , strlen(hello) , 0 ); 
    printf("Hello message sent\n"); 

}

static void runClient(GtkWidget *widget, gpointer data) { 
	
	int sock = 0, valread; 
    struct sockaddr_in serv_addr; 
    char *hello = (char*)std::string("Hello from client").c_str(); 
    char buffer[1024] = {0}; 
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) { 

        printf("\n Socket creation error \n"); 

    } else { 

	    serv_addr.sin_family = AF_INET; 
	    serv_addr.sin_port = htons(PORT); 
	       
	    // Convert IPv4 and IPv6 addresses from text to binary form 
//	    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)  
//	    { 
//	        printf("\nInvalid address/ Address not supported \n"); 
//	        return -1; 
//	    } else 
	    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) { 
	        printf("\nConnection Failed \n"); 
//	        return -1; 
	    } else { 
		    send(sock , hello , strlen(hello) , 0 ); 
		    printf("Hello message sent\n"); 
		    valread = read( sock , buffer, 1024); 
		    printf("%s\n",buffer ); 
	    }

    }

}

//****************************************************************************************************************
// Hello world GTK
//****************************************************************************************************************

static void print_hello(GtkWidget *widget, gpointer data) {

	// Basic Button Functionallity Example
	g_print ("Hello World\n");

}

void mat_to_array(cv::Mat& input, af::array& output) { 
	std::cout << "Starting Conversion\n"; 
	input.convertTo(input, CV_32FC3); 
	const unsigned w = input.cols; 
	const unsigned h = input.rows; 
	const unsigned size = h*w; 
	float r[size]; 
	float g[size]; 
	float b[size]; 		
	int tmp = 0; 
	for(unsigned i = 0; i < h; i++) { 
		for(unsigned j = 0; j < w; j++) { 
			cv::Vec3f ip = input.at<cv::Vec3f>(i,j); 
			tmp = j*h*i; 
			r[tmp] = ip[2]; 
			g[tmp] = ip[1]; 	
			b[tmp] = ip[0]; 					
		}
	}
	output = af::join(2, 
					  af::array(h,w,r), 
					  af::array(h,w,g),
					  af::array(h,w,b)
					  )/255.f; 
	std::cout << "Conversion Complete\n"; 
}

//****************************************************************************************************************
// Simple Open Camera Example Code
//****************************************************************************************************************

static void open_camera_with_DFT(GtkWidget *widget, gpointer data) { 

	// OpenCv Code Example

	std::cout << "Enter Camera to open: "; 
	int cam; 
	std::cin >> cam; 

	cv::VideoCapture stream1(cam);


	if (!stream1.isOpened()) { 
		std::cout << "cannot open camera";
	}

	cv::Mat I; 

	af::Window afWindow("Video Boi");
	af::array tmp; 

	while (!afWindow.close()) {

		stream1.read(I);
		mat_to_array(I,tmp); 
		afWindow.image(tmp);

		if (cv::waitKey(30) >= 0){ 
        	cv::destroyAllWindows(); 
			break;
        }


	}	

}

//****************************************************************************************************************
// Simple Open Camera Example Code
//****************************************************************************************************************

static void open_camera(GtkWidget *widget, gpointer data) { 

	// OpenCv Code Example

	std::cout << "Enter Camera to open: "; 
	int cam; 
	std::cin >> cam; 

	cv::VideoCapture stream1(cam);

	if (!stream1.isOpened()) { 
		std::cout << "cannot open camera";
	}

	cv::Mat cameraFrame;
	cv::Mat previousFrame; 
	cv::Mat difFrame; 

	while (true) {
		stream1.read(cameraFrame);
		cv::subtract(cameraFrame,previousFrame,difFrame); 
		cv::imshow("cam", difFrame);
		if (cv::waitKey(30) >= 0) { 
        	cv::destroyAllWindows(); 
			break;
		}
		previousFrame = cameraFrame; 
	}	

}

//****************************************************************************************************************
// DVideoToPictures
//****************************************************************************************************************

static void DVideoToPictures(GtkWidget *widget, gpointer data) { 

	// Setup stuff
	GtkWidget *D_window;
	GtkWidget *D_grid;
	std::string videoFile;

	// Window Setup
    D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "");

	FileSelector((GtkWindow*)D_window,videoFile); 

	std::cout << "File Path Selected: " << videoFile << "\n"; 

	std::string fileName = getFileNameFromPath(videoFile); 

	fileName = getWordsBeforeDelimiter(fileName,'.'); 

	std::cout << "File Selected: " << fileName << "\n"; 

	std::string picDir = std::string("/home/drake/Pictures/") + getFileNameFromPath(fileName) + std::string("_pic"); 

	std::string listPath = std::string("/home/drake/Lists/"); 

	if (boost::filesystem::exists(picDir)) {

		std::cout << "File Conversion Already Happened\n"; 

	} else { 


//ffmpeg -i input.avi -vcodec png output%05d.png

		std::cout << "Creating Directory: " << picDir << "\n"; 
		boost::filesystem::create_directory(picDir); 
		std::cout << "Filling Directory...\n"; 

		std::string cmd = std::string("ffmpeg -i ") 
						+ videoFile 
//						+ std::string(" -vf hue=s=0 ") 
						+ std::string(" -vcodec png ")
						+ picDir 
						+ std::string("/V%07d.png"); 

		std::cout << "Cmd: " << cmd << "\n"; 

		system(cmd.c_str()); 

		cmd = std::string("ls ") + picDir + std::string(" > ") + listPath + std::string("list_") + fileName; 

		std::cout << "Cmd: " << cmd << "\n"; 

		system(cmd.c_str()); 

	}

} 

std::string getFileNameFromPath(std::string path) { 

	int j = 0; 
	int i = 0; 
	for(std::string::iterator it = path.begin(); it != path.end(); ++it) { 
		j++; 
		if(*it == '/') { 
			i = j; 
		}
	}

	return path.substr(i); 

}

std::string getLastWordAfterDelimiter(std::string path, char d_lim) { 

	int j = 0; 
	int i = 0; 
	for(std::string::iterator it = path.begin(); it != path.end(); ++it) { 
		j++; 
		if(*it == d_lim) { 
			i = j; 
		}
	}

	return path.substr(i); 

}

std::string getWordsBeforeDelimiter(std::string path, char d_lim) { 

	int j = 0; 
	int i = 0; 
	for(std::string::iterator it = path.begin(); it != path.end(); ++it) { 
		j++; 
		if(*it == d_lim) { 
			i = j; 
		}
	}

	return path.substr(0,i-1); 

}

//****************************************************************************************************************
// DVideoProcessing
//****************************************************************************************************************

static void DVideoProcessing(GtkWidget *widget, gpointer data) { 

	std::cout << "Calling DVideoProcessing\n"; 

	// Setup stuff
	GtkWidget *D_window;
	GtkWidget *D_grid;

	GtkWidget *D_Open_Files_button;
	GtkWidget *D_Apply_Transform_button;
	GtkWidget *D_View_Images_button;	
	GtkWidget *D_Save_Images_button;		

	GtkWidget *D_Video_Select_Label;
	GtkWidget *D_Operator_Select_Label; 

	std::string listPath = std::string("/home/drake/Lists"); 

	std::vector<std::string> files; 

	//****************************************************************************************************************
	// Default Setup
	//****************************************************************************************************************

	// Window Setup
    D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "D Video Processing");

    // Grid Setup
	D_grid = gtk_grid_new ();
	gtk_container_add (GTK_CONTAINER (D_window), D_grid);

	D_Video_Select_Label = gtk_label_new ("Select Video to Edit: ");
	gtk_grid_attach (GTK_GRID (D_grid), D_Video_Select_Label, 0, 0, 1, 1);	

	vlistFilesInDirectory(listPath, files); 

	D_Video_Select_Combo = gtk_combo_box_text_new();

	for (std::vector<std::string>::iterator it = files.begin() ; it != files.end(); ++it) { 
		gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Video_Select_Combo), NULL, (const char *)((*it).c_str()));
		std::cout << "File: " << *it << "\n"; 
	} 

//    gtk_combo_box_set_active(GTK_COMBO_BOX(D_Video_Select_Combo), 0);
//    gtk_combo_box_set_active(GTK_COMBO_BOX(D_Video_Transform_Select_Combo), 0);
	gtk_grid_attach (GTK_GRID (D_grid), D_Video_Select_Combo, 1, 0, 1, 1);
	g_signal_connect (D_Video_Select_Combo, "changed", G_CALLBACK(read_Combo), D_Video_Select_Combo );

	D_Operator_Select_Label = gtk_label_new ("Select Operator: ");
	gtk_grid_attach (GTK_GRID (D_grid), D_Operator_Select_Label, 0, 1, 1, 1);	

    D_Video_Transform_Select_Combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Video_Transform_Select_Combo), NULL, "pairWiseDiff");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Video_Transform_Select_Combo), NULL, "norm");		    
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Video_Transform_Select_Combo), NULL, "realDFT");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Video_Transform_Select_Combo), NULL, "probRep");

	gtk_grid_attach (GTK_GRID (D_grid), D_Video_Transform_Select_Combo, 1, 1, 1, 1);
	g_signal_connect (D_Video_Transform_Select_Combo, "changed", G_CALLBACK(read_Combo), D_Video_Transform_Select_Combo );

	D_Open_Files_button = gtk_button_new_with_label ("Open Files");
	g_signal_connect (D_Open_Files_button, "clicked", G_CALLBACK (import_Video_Picture_List), NULL);
	g_signal_connect_swapped (D_Open_Files_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_Open_Files_button, 0, 2, 1, 1);

	D_Apply_Transform_button = gtk_button_new_with_label ("Apply Transform");
	g_signal_connect (D_Apply_Transform_button, "clicked", G_CALLBACK (DApplyTransform), NULL);
	g_signal_connect_swapped (D_Apply_Transform_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_Apply_Transform_button, 0, 3, 1, 1);

	if(open_File_Trigger) { 
		D_View_Images_button = gtk_button_new_with_label ("View Images");
		g_signal_connect (D_View_Images_button, "clicked", G_CALLBACK (DViewImages), NULL);
		g_signal_connect_swapped (D_View_Images_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
		gtk_grid_attach (GTK_GRID (D_grid), D_View_Images_button, 0, 4, 1, 1);

		D_Save_Images_button = gtk_button_new_with_label ("Save Images");
		g_signal_connect (D_Save_Images_button, "clicked", G_CALLBACK (D_Save_Images), NULL);
		g_signal_connect_swapped (D_Save_Images_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
		gtk_grid_attach (GTK_GRID (D_grid), D_Save_Images_button, 0, 5, 1, 1);		
	}

	if(apply_Transform_Trigger && open_File_Trigger) { 
		std::cout << "Apply Transform\n"; 
	}

	gtk_widget_show_all (D_window);

}

static void DViewImages(GtkWidget *widget, gpointer data) { 
	std::cout << "Opening Images\n";
    af::Window afWindow("Training Set Inspector");
    int input; 

    while(!afWindow.close()) { 

    	for(std::vector<af::array>::iterator it = images.begin(); it != images.end(); ++it) { 
	        afWindow.image(*it);
    	}

    	std::cout << "Continue?\n"; 
    	std::cin >> input; 
    	if(input == 0) { 
    		break; 
    	} 

	}

	DVideoProcessing(NULL,NULL); 
}

static void import_Video_Picture_List(GtkWidget *widget, gpointer &data) { 
//	static GtkWidget *D_Video_Select_Combo; 
//	static GtkWidget *D_Operator_Select_Combo; 
    const char * val = gtk_combo_box_text_get_active_text ((GtkComboBoxText*)D_Video_Select_Combo);
    std::fstream file; 
    std::string line; 

	std::string path = std::string("/home/drake/Pictures/") + getLastWordAfterDelimiter(val, '_') + "_pic/"; 

    std::cout << "Video Pictures to Import: " << val << "\n"; 

	std::string listPath = std::string("/home/drake/Lists/") + std::string(val); 

	file.open(listPath); 
	if (file.is_open()) {
		while ( getline (file,line)) {
			std::cout << (const char*)(path + line).c_str() << "\n";

			images.push_back(af::loadImageNative( (const char*)(path + line).c_str() )); 
		}
		file.close();
	} else { 
		std::cout << "File would not open\n"; 
	}

	open_File_Trigger = true; 

	std::cout << "Pictures Opening Complete.\n"; 

	DVideoProcessing(NULL,NULL); 

}

static void DApplyTransform(GtkWidget *widget, gpointer &data) { 

    const char * val = gtk_combo_box_text_get_active_text ((GtkComboBoxText*)D_Video_Transform_Select_Combo);
    std::cout << "val: " << val << "\n"; 
    if(strcmp(val,"pairWiseDiff") == 0) { 
    	std::cout << "Running: pairwiseDifference\n"; 
		pairwiseDifference(images); 
    } else if(strcmp(val, "norm") == 0) { 

    } else if(strcmp(val, "realDFT") == 0) { 

    } else if(strcmp(val, "probRep") == 0) { 
    	sequentialProbabilityRepresentation(images); 
    } else { 

    }

	DVideoProcessing(NULL,NULL); 
	
}

static void D_Save_Images(GtkWidget *widget, gpointer data) { 

	D_update_time(); 


	const char * val = gtk_combo_box_text_get_active_text ((GtkComboBoxText*)D_Video_Select_Combo);
	std::string directName = getLastWordAfterDelimiter(val, '_');  

	std::string imagesDirectory = "/home/drake/Pictures/" + directName + "-" + current_Time + "_pic/";

	if (boost::filesystem::exists(imagesDirectory)) {

		std::cout << "File Conversion Already Happened\n"; 

	} else { 

		std::string cmd = std::string("mkdir ") + imagesDirectory; 

		std::cout << "CMD: " << cmd << "\n"; 

 		system((const char*)cmd.c_str()); 

		std::string fileName; 
		int c = 1; 

		for(auto it = images.begin(); it != images.end(); ++it) { 
			std::ostringstream counter;
			counter << std::setw(7) << std::setfill('0') << c;
			std::cout << counter.str() << std::endl;
			std::cout << "File Path: " << imagesDirectory + counter.str() + std::string(".png") << "\n"; 
			fileName = imagesDirectory + counter.str() + std::string(".png"); 
			af::saveImageNative(fileName.c_str(),*it); 
			c++; 
		}

		cmd = std::string("ls ") + imagesDirectory + std::string(" > /home/drake/Lists/list_") + directName + std::string("-") + current_Time; 

		system((const char*)cmd.c_str()); 

	}


}

//****************************************************************************************************************
// DCAD Start point
//****************************************************************************************************************

static void _3dcad(GtkWidget *widget, gpointer data) { 

	// Setup stuff
	GtkWidget *D_window;
	GtkWidget *D_grid;

	GtkWidget *D_Open_Heiarchey_button;
	GtkWidget *D_New_Heiarchey_button;
	GtkWidget *D_HardCoded_Heiarchey_button;

	GtkWidget *D_Input_Heiarchey_Field; 

	// Window Setup
    D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "DCAD");

    // Grid Setup
	D_grid = gtk_grid_new ();
	gtk_container_add (GTK_CONTAINER (D_window), D_grid);

	/*
		// Text view setup
		GtkWidget *D_Heiarcheys;
		GtkTextBuffer *D_Heiarcheys_Buffer;

		D_Heiarcheys = gtk_text_view_new ();
		D_Heiarcheys_Buffer = gtk_text_view_get_buffer (GTK_TEXT_VIEW (D_Heiarcheys));
		const gchar* D_Heiarcheys_Buffer_data = (const gchar*)((slistFilesInDirectory("DCAD/SAVEDHEIARCHEYS/")).c_str()); 
		gtk_text_buffer_set_text (D_Heiarcheys_Buffer, D_Heiarcheys_Buffer_data, -1);
		gtk_grid_attach (GTK_GRID (D_grid), D_Heiarcheys, 0, 0, 1, 1);
	*/ 

	// Open Hierarchy Document, create a new Hierarchy Document, or select from hard coded Hierarchys
	#ifdef __DIALOG__
		ShowMessage("Open Hierarchy Document\nCreate a new Hierarchy Document\nSelect from hard coded Hierarchys\n","Welcome to DCAD"); 
	#endif

    // D_Open_Heiarchey_button Specific Code
	D_Open_Heiarchey_button = gtk_button_new_with_label ("Open Heiarchey");
	g_signal_connect (D_Open_Heiarchey_button, "clicked", G_CALLBACK (Open_Heiarchey), NULL);
	g_signal_connect_swapped (D_Open_Heiarchey_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_Open_Heiarchey_button, 0, 0, 1, 1);

    // D_New_Heiarchey_button Specific Code
	D_New_Heiarchey_button = gtk_button_new_with_label ("New Heiarchey");
	g_signal_connect (D_New_Heiarchey_button, "clicked", G_CALLBACK (New_Heiarchey), NULL);
	g_signal_connect_swapped (D_New_Heiarchey_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_New_Heiarchey_button, 1, 0, 1, 1);

    // D_HardCoded_Heiarchey_button Specific Code
	D_HardCoded_Heiarchey_button = gtk_button_new_with_label ("HardCoded Heiarchey");
	g_signal_connect (D_HardCoded_Heiarchey_button, "clicked", G_CALLBACK (HardCoded_Heiarchey), NULL);
	g_signal_connect_swapped (D_HardCoded_Heiarchey_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_HardCoded_Heiarchey_button, 2, 0, 1, 1);		

	// Allocate Memory and Fill the mesh Hieraarchy array 

	// Display stuff
	gtk_widget_show_all (D_window);

	// Run DCAD loop
	std::cout << "Start DCAD\n"; 

}

// https://developer.gnome.org/gtk3/stable/GtkFileChooserDialog.html

static void Open_Heiarchey(GtkWidget *widget, gpointer *data) { 

	// Setup stuff
	GtkWidget *D_window;
	GtkWidget *D_grid;

	// Window Setup
    D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "DCAD");

    // Grid Setup
	D_grid = gtk_grid_new ();
	gtk_container_add (GTK_CONTAINER (D_window), D_grid);

	FileChooser((GtkWindow*) D_window, default_open_file_function ); 

	// Display stuff
	gtk_widget_show_all (D_window);

}

static void New_Heiarchey(GtkWidget *widget, gpointer *data) { 

	// List available meshes
	// Add meshes to fifo
	// Save fifo as Heiarchey
	// Draw Heiarchey

}

static void HardCoded_Heiarchey(GtkWidget *widget, gpointer *data) { 
	// List available Heiarcheys

}

static void Open_Mesh(GtkWidget *widget, gpointer *data) { 
	// List available Meshes

	// Setup stuff
	GtkWidget *D_window;
	GtkWidget *D_grid;

	// Window Setup
    D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "DCAD");

    // Grid Setup
	D_grid = gtk_grid_new ();
	gtk_container_add (GTK_CONTAINER (D_window), D_grid);

	FileChooser((GtkWindow*) D_window, default_open_file_function ); 

	// Display stuff
	gtk_widget_show_all (D_window);

}

static void New_Mesh(GtkWidget *widget, gpointer *data) { 

}

static void HardCoded_Mesh(GtkWidget *widget, gpointer *data) { 
	// List available meshes

}

//****************************************************************************************************************
// Neural Net Architect Start point
//****************************************************************************************************************

static void construct_Neural_Net(GtkWidget *widget, gpointer data) { 

	std::cout << "==================================================\n";
	std::cout << "Construct FCO Page 1\n"; 
	std::cout << "==================================================\n";

	// Setup stuff
	GtkWidget *D_window;
	GtkWidget *D_grid;

	GtkWidget *D_Construct_button;

	GtkWidget *D_Operator_Count_Label;
	GtkWidget *D_Operator_Name_Label;	
	GtkWidget *D_Operator_Select_Label;
	GtkWidget *D_Operator_Dimension_Label;

//	GtkTextBuffer *D_Operator_Count_View_Buffer;
	GtkTextBuffer *D_Operator_Select_view_Buffer;
	GtkTextBuffer *D_Operator_Name_view_Buffer;	
	GtkTextBuffer *D_Operator_Dimension_view_Buffer;	

//	GtkWidget *D_Operator_Select_Combo; 
	GtkWidget *D_Operator_Name_Field; 	
	GtkWidget *D_Operator_Dimension_Field; 	
	GtkWidget *D_Operator_Count_Input_Field; 

	std::string val; 

	//****************************************************************************************************************
	// Default Setup
	//****************************************************************************************************************

	// Window Setup
    D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "Neural Net Contructor");

    // Grid Setup
	D_grid = gtk_grid_new ();
	gtk_container_add (GTK_CONTAINER (D_window), D_grid);

	//****************************************************************************************************************
	// Operator Count Control
	//****************************************************************************************************************

	D_Operator_Count_Label = gtk_label_new ("Enter Operator Count: ");
	gtk_grid_attach (GTK_GRID (D_grid), D_Operator_Count_Label, 0, 0, 1, 1);		

	// Operator Count Input Field
	D_Operator_Count_Input_Field = gtk_entry_new_with_buffer (D_Operator_Count_Buffer);
	gtk_grid_attach (GTK_GRID (D_grid), D_Operator_Count_Input_Field, 1, 0, 1, 1);	
	g_signal_connect (D_Operator_Count_Input_Field, "changed", G_CALLBACK (construct_Neural_Net), NULL);		
	g_signal_connect_swapped (D_Operator_Count_Input_Field, "changed", G_CALLBACK (gtk_widget_destroy), D_window);	

	opCount = std::atoi(gtk_entry_buffer_get_text( (GtkEntryBuffer *) D_Operator_Count_Buffer) ); 

	if(opCount > 0) { 

		//****************************************************************************************************************
		// Reaction to non-zero operator count. 
		//****************************************************************************************************************

		std::cout << "Op Count: " << opCount << "\n"; 

		D_Operator_Name_Buffers = NULL; 
		D_Operator_Dimension_Buffers = NULL; 

//		D_Operator_Name_Buffers = (GtkEntryBuffer **)realloc(D_Operator_Name_Buffers, opCount*sizeof(GtkEntryBuffer*));
//		D_Operator_Dimension_Buffers = (GtkEntryBuffer **)realloc(D_Operator_Dimension_Buffers, (opCount+1)*sizeof(GtkEntryBuffer*));

		D_Operator_Name_Buffers = (GtkEntryBuffer **)malloc(opCount*sizeof(gtk_entry_buffer_new (NULL,-1)));
		D_Operator_Dimension_Buffers = (GtkEntryBuffer **)malloc((opCount+1)*sizeof(gtk_entry_buffer_new (NULL,-1)));
		D_Operator_Select_Combo = (GtkWidget **)malloc(opCount*sizeof(gtk_combo_box_text_new()));

		D_Buffer_Collection = (GtkWidget ***)malloc(sizeof(D_Operator_Name_Buffers) + sizeof(D_Operator_Dimension_Buffers) + sizeof(D_Operator_Select_Combo)); 

		D_Buffer_Collection[0] = (GtkWidget**)D_Operator_Name_Buffers; 
		D_Buffer_Collection[1] = (GtkWidget**)D_Operator_Dimension_Buffers; 
		D_Buffer_Collection[2] = D_Operator_Select_Combo; 

		//****************************************************************************************************************
		// Request Name // Need to set default
		//****************************************************************************************************************

		for(int i = 0; i < opCount; i++) { 

			D_Operator_Name_Buffers[i] = gtk_entry_buffer_new (NULL,-1);
			D_Operator_Dimension_Buffers[i] = gtk_entry_buffer_new (NULL,-1);
//			D_Operator_Select_Combo[i] = gtk_combo_box_text_new();

			val = "T_" + std::to_string(i); 
			gtk_entry_buffer_set_text (D_Operator_Name_Buffers[i],(char *)val.c_str(),-1); 

		}

		D_Operator_Dimension_Buffers[opCount] = gtk_entry_buffer_new (NULL,-1);

		for(int i = 0; i < opCount; i++) { 

//			add_View(D_Operator_Name_view,D_Operator_Name_view_Buffer,D_grid,(char *)"Enter Operator Name",0,4*i+1); 

			D_Operator_Name_Label = gtk_label_new ("Enter Operator Count: ");
			gtk_grid_attach (GTK_GRID (D_grid), D_Operator_Name_Label, 0,4*i+1, 1, 1);		

			// Operator Name View Input Field
			D_Operator_Name_Field = gtk_entry_new_with_buffer (D_Operator_Name_Buffers[i]);
			gtk_grid_attach (GTK_GRID (D_grid), D_Operator_Name_Field, 1, 4*i+1, 1, 1);		

			val = "Enter Input Dimension: X_" + std::to_string(i); 

			std::cout << val << "\n"; 

			D_Operator_Dimension_Label = gtk_label_new ((char *)val.c_str());
			gtk_grid_attach (GTK_GRID (D_grid), D_Operator_Dimension_Label, 0,4*i+2, 1, 1);		

			// Input Field
			D_Operator_Dimension_Field = gtk_entry_new_with_buffer (D_Operator_Dimension_Buffers[i]);
			gtk_grid_attach (GTK_GRID (D_grid), D_Operator_Dimension_Field, 1, 4*i+2, 1, 1);		

			val = "Enter Output Dimension: X_" + std::to_string(i+1) ; 

			std::cout << val << "\n"; 
			D_Operator_Dimension_Label = gtk_label_new ((char *)val.c_str());
			gtk_grid_attach (GTK_GRID (D_grid), D_Operator_Dimension_Label, 0,4*i+3, 1, 1);		

			// Input Field
			D_Operator_Dimension_Field = gtk_entry_new_with_buffer (D_Operator_Dimension_Buffers[i+1]);
			gtk_grid_attach (GTK_GRID (D_grid), D_Operator_Dimension_Field, 1, 4*i+3, 1, 1);					

			D_Operator_Select_Label = gtk_label_new ("Select Operator: ");
			gtk_grid_attach (GTK_GRID (D_grid), D_Operator_Select_Label, 0,4*i+4, 1, 1);		

			// Create Combo box
		    D_Operator_Select_Combo[i] = gtk_combo_box_text_new();
		    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Operator_Select_Combo[i]), NULL, "Matrix Multiplication");
		    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Operator_Select_Combo[i]), NULL, "Vector Translation");		    
		    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Operator_Select_Combo[i]), NULL, "Arc-tangent");
		    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Operator_Select_Combo[i]), NULL, "Heaviside step function");
		    gtk_combo_box_set_active(GTK_COMBO_BOX(D_Operator_Select_Combo[i]), 1);
			gtk_grid_attach (GTK_GRID (D_grid), D_Operator_Select_Combo[i], 1, 4*i+4, 1, 1);
			g_signal_connect (D_Operator_Select_Combo[i], "changed", G_CALLBACK(read_Combo), D_Operator_Select_Combo[i] );

		}

		// Construct Button
		D_Construct_button = gtk_button_new_with_label("Construct Neural Net");
		g_signal_connect (D_Construct_button, "clicked", G_CALLBACK (create_Active_Neural_Net), NULL);
		g_signal_connect_swapped (D_Construct_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
		gtk_grid_attach (GTK_GRID (D_grid), D_Construct_button, 0, 4*opCount+4, 1, 1);

	}

	// Display stuff
	gtk_widget_show_all (D_window);

}

static void create_Active_Neural_Net(GtkWidget *widget, gpointer *data) { 

	std::cout << "==================================================\n";
	std::cout << "Construct FCO Page 2\n"; 
	std::cout << "==================================================\n";

	Neural_Net->setOperatorCount(opCount); 
	Neural_Net->allocateOperatorSpace(); 
	Neural_Net->allocateDimSpace(); 

	std::string name; 
	char * dim_I; 
	char * dim_O; 
	std::string opp;  	

	for(int i = 0; i < opCount; i++) { 

		name = std::string(gtk_entry_buffer_get_text( (GtkEntryBuffer *) D_Buffer_Collection[0][i] ) ) ; 
		dim_I = (char *)gtk_entry_buffer_get_text( (GtkEntryBuffer *) D_Buffer_Collection[1][i] ); 
		dim_O = (char *)gtk_entry_buffer_get_text( (GtkEntryBuffer *) D_Buffer_Collection[1][i+1] ); 		
		opp = std::string(gtk_combo_box_text_get_active_text ((GtkComboBoxText*)D_Buffer_Collection[2][i]) ); 

		std::cout << "Name: " << name << "\n"; 
		std::cout << "dim(X_" << i << ") = " << dim_I << "\n"; 
		std::cout << "dim(X_" << i+1 << ") = " << dim_O << "\n"; 		
	    std::cout << "Operation: " << opp << "\n\n"; 

	    Neural_Net->setSpaceDim(i,atoi(dim_I));
	    Neural_Net->setSpaceDim(i+1,atoi(dim_O));	     
	    std::cout << "Dim Set\n"; 

	    if(opp == "Matrix Multiplication") { 
		    Neural_Net->setOperator( i , matrixOperator, opp ); 
		    std::cout << "MM Opp Set\n"; 
	    } else if(opp == "Vector Translation") { 
		    Neural_Net->setOperator( i , additionOperator, opp ); 
		    std::cout << "VT Opp Set\n"; 
	    } else if(opp == "Arc-tangent") { 
		    Neural_Net->setOperator( i , nonLinearOperator_arcTan, opp ); 
		    std::cout << "AT Opp Set\n"; 
	    }

	}

	#ifdef __GENERAL_DEBUG__
		std::cout << "Preparing to call Data Allocation\n"; 
	#endif

	Neural_Net->allocateKernels(); 

	Neural_Net->debugKernels(); 

	#ifdef __GENERAL_DEBUG__
		std::cout << "Finite Composition Operator Created\n\n"; 
	#endif
	
	select_Training_Set(); 

}

static void select_Training_Set() { 

	std::cout << "==================================================\n";
	std::cout << "Construct FCO Page 3\n"; 
	std::cout << "==================================================\n";

	// Setup stuff
	GtkWidget *D_window;
	GtkWidget *D_grid;

	GtkWidget *D_Training_Set_Select_Label;
	GtkWidget *D_Training_Set_Select_Combo;  

	GtkWidget *D_Training_Algorithm_Select_Label;
	GtkWidget *D_Training_Algorithm_Select_Combo;  

	GtkWidget *D_Training_Preprocessing_Select_Label;
	GtkWidget *D_Training_Preprocessing_Select_Combo;  		

	GtkWidget *D_Images_Per_Catagory_Label;
	GtkWidget *D_Images_Per_Catagory_Field;	 	
	GtkEntryBuffer * D_Images_Per_Catagory_Field_Buffer;  

	GtkWidget *D_Start_Training_button;

	//****************************************************************************************************************
	// Default Setup
	//****************************************************************************************************************

	// Window Setup
    D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "Training Set and Algorithm Selector");

    // Grid Setup
	D_grid = gtk_grid_new ();
	gtk_container_add (GTK_CONTAINER (D_window), D_grid);

	//****************************************************************************************************************
	// View Setup
	//****************************************************************************************************************

	D_Training_Set_Select_Label = gtk_label_new ("Selet Training Set: ");
	gtk_grid_attach (GTK_GRID (D_grid), D_Training_Set_Select_Label, 0, 0, 1, 1);

	// Training Set Select
	// Create Combo box
    D_Training_Set_Select_Combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Training_Set_Select_Combo), NULL, "Minst Character Set");
    gtk_combo_box_set_active(GTK_COMBO_BOX(D_Training_Set_Select_Combo), 0);
	gtk_grid_attach (GTK_GRID (D_grid), D_Training_Set_Select_Combo, 1, 0, 1, 1);
	g_signal_connect (D_Training_Set_Select_Combo, "changed", G_CALLBACK(read_Combo), D_Training_Set_Select_Combo );

	D_Training_Algorithm_Select_Label = gtk_label_new ("Selet Training Algorithm: ");
	gtk_grid_attach (GTK_GRID (D_grid), D_Training_Algorithm_Select_Label, 0, 1, 1, 1);

	// Training Algorithm Select
	// Create Combo box
    D_Training_Algorithm_Select_Combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Training_Algorithm_Select_Combo), NULL, "Random Trials");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Training_Algorithm_Select_Combo), NULL, "Gradient Descent");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Training_Algorithm_Select_Combo), NULL, "Sub-gradient Method");    
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Training_Algorithm_Select_Combo), NULL, "Newton's Method");
    gtk_combo_box_set_active(GTK_COMBO_BOX(D_Training_Algorithm_Select_Combo), 0);
	gtk_grid_attach (GTK_GRID (D_grid), D_Training_Algorithm_Select_Combo, 1, 1, 1, 1);
	g_signal_connect (D_Training_Algorithm_Select_Combo, "changed", G_CALLBACK(read_Combo), D_Training_Algorithm_Select_Combo );

	D_Training_Preprocessing_Select_Label = gtk_label_new ("Selet Training Preprocessing: ");
	gtk_grid_attach (GTK_GRID (D_grid), D_Training_Preprocessing_Select_Label, 0, 2, 1, 1);

	// Preprocessing Select
	// Create Combo box
    D_Training_Preprocessing_Select_Combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Training_Preprocessing_Select_Combo), NULL, "None");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(D_Training_Preprocessing_Select_Combo), NULL, "Remove Outliers");    
    gtk_combo_box_set_active(GTK_COMBO_BOX(D_Training_Preprocessing_Select_Combo), 0);
	gtk_grid_attach (GTK_GRID (D_grid), D_Training_Preprocessing_Select_Combo, 1, 2, 1, 1);
	g_signal_connect (D_Training_Preprocessing_Select_Combo, "changed", G_CALLBACK(read_Combo),D_Training_Preprocessing_Select_Combo );		

	D_Images_Per_Catagory_Label = gtk_label_new ("Number of Samples Per Catagory: ");
	gtk_grid_attach (GTK_GRID (D_grid), D_Images_Per_Catagory_Label, 0, 3, 1, 1);

	// Input Field
	D_Images_Per_Catagory_Field_Buffer = gtk_entry_buffer_new (NULL,-1);
	D_Images_Per_Catagory_Field = gtk_entry_new_with_buffer (D_Images_Per_Catagory_Field_Buffer);
	
	gtk_grid_attach (GTK_GRID (D_grid), D_Images_Per_Catagory_Field, 1, 3, 1, 1);

	int size = sizeof(D_Training_Set_Select_Combo) 
			 + sizeof(D_Training_Algorithm_Select_Combo) 
			 + sizeof(D_Training_Preprocessing_Select_Combo) 
			 + sizeof(D_Images_Per_Catagory_Field_Buffer);  

	D_Training_Setup_Data = (GtkWidget **)malloc(size); 

	D_Training_Setup_Data[0] = (GtkWidget *)D_Training_Set_Select_Combo; 
	D_Training_Setup_Data[1] = (GtkWidget *)D_Training_Algorithm_Select_Combo; 
	D_Training_Setup_Data[2] = (GtkWidget *)D_Training_Preprocessing_Select_Combo; 
	D_Training_Setup_Data[3] = (GtkWidget *)D_Images_Per_Catagory_Field_Buffer; 

    // Hello Button Specific Code
	D_Start_Training_button = gtk_button_new_with_label ("Start Training");
	g_signal_connect (D_Start_Training_button, "clicked", G_CALLBACK (start_Training), NULL);
	g_signal_connect_swapped (D_Start_Training_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_Start_Training_button, 0, 4, 1, 1);

	//****************************************************************************************************************
	// Open Window
	//****************************************************************************************************************

	// Display stuff
	gtk_widget_show_all (D_window);

}

static void start_Training(GtkWidget *widget, gpointer data) {  

	std::cout << "==================================================\n";
	std::cout << "Construct FCO Page 4\n"; 
	std::cout << "==================================================\n";

	std::cout << "Starting Training Process\n"; 

    const char * D_Training_Set_Name = gtk_combo_box_text_get_active_text ((GtkComboBoxText*)D_Training_Setup_Data[0]);
    std::cout << "Training Set Selected: " << D_Training_Set_Name << "\n"; 

    const char * D_Training_Algorithm_Name = gtk_combo_box_text_get_active_text ((GtkComboBoxText*)D_Training_Setup_Data[1]);
    std::cout << "Training Algorithm Selected: " << D_Training_Algorithm_Name << "\n";     
	Neural_Net->setInformationAboutTrainingAlgorithm(std::string(D_Training_Algorithm_Name)); 

    const char * D_Training_Preprocessing_Name = gtk_combo_box_text_get_active_text ((GtkComboBoxText*)D_Training_Setup_Data[2]);
    std::cout << "Training Preprocessing Selected: " << D_Training_Preprocessing_Name << "\n";     
    Neural_Net->setInformationAboutPreprocessingAlgorithm(std::string(D_Training_Preprocessing_Name)); 

	int D_Training_Count = atoi(gtk_entry_buffer_get_text((GtkEntryBuffer *)D_Training_Setup_Data[3]));
    std::cout << "Number of Images Per Catagory: " << D_Training_Count << "\n\n";

    // Import Training Sets

    if(strcmp(D_Training_Set_Name, "Minst Character Set") == 0) { 
    	std::cout << "import_Minst_Dataset\n"; 
		Neural_Net->setInformationAboutTrainingSpace("Minst Character Set"); 
		import_Minst_Dataset(D_Training_Count); 
    } else { 
    	std::cout << "Error Importing Training Set.\n"; 
    }

    #ifdef __GENERAL_DEBUG__
    	std::cout << "inspectTrainingSet\n"; 
		Neural_Net->inspectTrainingSet(); 
	#endif

	// Do Preprocessing

	if(strcmp(D_Training_Algorithm_Name, "None") == 0) { 

	} else if(strcmp(D_Training_Algorithm_Name,"Remove Outliers") == 0) { 
		Neural_Net->remove_Outliers(); 
	} else { 

	}

	// Do Training	

	if(strcmp(D_Training_Algorithm_Name, "Gradient Descent") == 0) { 
		Neural_Net->gradient_Descent(); 
	} else if(strcmp(D_Training_Algorithm_Name, "Sub-gradient Method") == 0 )  { 
		Neural_Net->sub_Gradient_Method();
	} else if(strcmp(D_Training_Algorithm_Name, "Random Trials") == 0) { 
		Neural_Net->random_Trials(); 
	} else { 
		std::cout << "Error Selecting Training Algorithm\n"; 
	}

	std::cout << "Training is complete\n"; 

	Neural_Net->generateDocumentation(); 

}

//****************************************************************************************************************
// DSOFT Start Point
//****************************************************************************************************************

void DFSM::activate(GtkApplication *app, gpointer user_data) {

	GtkWidget *D_hello_button;
	GtkWidget *D_web_camera_button;
	GtkWidget *D_3DCad_button;	
	GtkWidget *D_construct_Neural_Net_button; 
	GtkWidget *D_open_image_button; 
	GtkWidget *D_open_website_button;
	GtkWidget *D_Create_Report_Button; 
	GtkWidget *D_VideoToPictures_Button;
	GtkWidget *D_VideoProcessing_Button;
	GtkWidget *D_Client_Server_Pair_Button; 

	#ifdef __DIALOG__
		ShowMessage("","Welcome to DSOFT"); 
	#endif

	TestWidget = gtk_application_window_new(app);

	// Window Setup
    GtkWidget * D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "DCAD");

    // Grid Setup
	GtkWidget *D_grid = gtk_grid_new ();
	gtk_container_add (GTK_CONTAINER (D_window), D_grid);

    // Hello Button Specific Code
	D_hello_button = gtk_button_new_with_label ("Hello World");
	g_signal_connect (D_hello_button, "clicked", G_CALLBACK (print_hello), NULL);
	g_signal_connect_swapped (D_hello_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_hello_button, 0, 0, 1, 1);

    // Web Camera Button Specific Code
	D_web_camera_button = gtk_button_new_with_label ("Open Web Camera");
	g_signal_connect (D_web_camera_button, "clicked", G_CALLBACK (open_camera), NULL);
	g_signal_connect_swapped (D_web_camera_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_web_camera_button, 0, 1, 1, 1);

    // 3D Cad Button Specific Code
	D_3DCad_button = gtk_button_new_with_label ("Open 3D Cad ");
	g_signal_connect (D_3DCad_button, "clicked", G_CALLBACK (_3dcad), NULL);
	g_signal_connect_swapped (D_3DCad_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_3DCad_button, 0, 2, 1, 1);

    // Neural Net Constructor Button Code
	D_construct_Neural_Net_button = gtk_button_new_with_label ("Construct Neural Net");
	g_signal_connect (D_construct_Neural_Net_button, "clicked", G_CALLBACK (construct_Neural_Net), NULL);
	g_signal_connect_swapped (D_construct_Neural_Net_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_construct_Neural_Net_button, 0, 3, 1, 1);

    // Open Image Constructor Button Code
	D_open_image_button = gtk_button_new_with_label ("Open Image");
	g_signal_connect (D_open_image_button, "clicked", G_CALLBACK (open_image), NULL);
	g_signal_connect_swapped (D_open_image_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_open_image_button, 0, 4, 1, 1);

    // Open Website Constructor Button Code
	D_open_website_button = gtk_button_new_with_label ("Open Website");
	g_signal_connect (D_open_website_button, "clicked", G_CALLBACK (open_website), NULL);
	g_signal_connect_swapped (D_open_website_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_open_website_button, 0, 4, 1, 1);

    // Create Experiment Report
	D_Create_Report_Button = gtk_button_new_with_label ("Create Experiment Report");
	g_signal_connect (D_Create_Report_Button, "clicked", G_CALLBACK (create_experiment_report), NULL);
	g_signal_connect_swapped (D_Create_Report_Button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_Create_Report_Button, 0, 5, 1, 1);	

    // Video To Pictures Specific Code
	D_VideoToPictures_Button = gtk_button_new_with_label ("Video To Pictures");
	g_signal_connect (D_VideoToPictures_Button, "clicked", G_CALLBACK (DVideoToPictures), NULL);
	g_signal_connect_swapped (D_VideoToPictures_Button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_VideoToPictures_Button, 0, 6, 1, 1);

    // Video Processing Specific Code
	D_VideoProcessing_Button = gtk_button_new_with_label ("Video Processing");
	g_signal_connect (D_VideoProcessing_Button, "clicked", G_CALLBACK (DVideoProcessing), NULL);
	g_signal_connect_swapped (D_VideoProcessing_Button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_VideoProcessing_Button, 0, 7, 1, 1);

    // Client Server Pair Specific Code
	D_Client_Server_Pair_Button = gtk_button_new_with_label ("Client Server Pair");
	g_signal_connect (D_Client_Server_Pair_Button, "clicked", G_CALLBACK (D_Client_Server_Pair), NULL);
	g_signal_connect_swapped (D_Client_Server_Pair_Button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach (GTK_GRID (D_grid), D_Client_Server_Pair_Button, 0, 8, 1, 1);
	

	// Open Window
	gtk_widget_show_all (D_window);

}

//****************************************************************************************************************
// Open Image Sequence
//****************************************************************************************************************

static void open_image(GtkWidget *widget, gpointer *data) { 

    GtkWidget *D_window;

	// Window Setup
    D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "");

	FileChooser( (GtkWindow*)D_window, display_image ); 

}

void display_image(char * filename) { 

    GtkWidget *D_window;
	GtkWidget *D_grid;
	GtkWidget *image; 

	// Window Setup
    D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "DCAD");

    // Grid Setup
	D_grid = gtk_grid_new ();
	gtk_container_add (GTK_CONTAINER (D_window), D_grid);

	image = gtk_image_new_from_file (filename);

	gtk_grid_attach (GTK_GRID (D_grid), image, 0, 0, 1, 1);

	gtk_widget_show_all (D_window);

}

//****************************************************************************************************************
// DBROWSER
//****************************************************************************************************************

static void open_website(GtkWidget *widget, gpointer *data) { 

	// Setup stuff
	GtkWidget *D_window;
	GtkWidget *D_grid;

	GtkWidget *D_Open_button;

	GtkWidget *D_Website_Select_View;
	GtkTextBuffer *D_Website_Select_View_Buffer;

	GtkWidget *D_Website_Select_Field;

	GtkWidget *D_Html_Field;
	GtkEntryBuffer *D_Html_Field_Buffer;

	// Window Setup
    D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "DCAD");

    // Grid Setup
	D_grid = gtk_grid_new ();
	gtk_container_add (GTK_CONTAINER (D_window), D_grid);

	// Website Select View
	add_View(D_Website_Select_View,D_Website_Select_View_Buffer,D_grid,(char *)"Enter Website Name",0,0); 

	// Html Field
//	D_Html_Field = gtk_entry_new_with_buffer(D_Html_Field_Buffer);
	D_Website_Select_Field = gtk_entry_new_with_buffer(D_Website_Select_Field_Buffer); 
	gtk_grid_attach (GTK_GRID (D_grid), D_Website_Select_Field, 1, 0, 1, 1);	

	// Open Button
	D_Open_button = gtk_button_new_with_label ("Open Website");
	gtk_grid_attach (GTK_GRID (D_grid), D_Open_button, 0, 1, 1, 1);
	g_signal_connect (D_Open_button, "clicked", G_CALLBACK (open_website), NULL);		
	g_signal_connect_swapped (D_Open_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);

	D_Html_Field_Buffer = gtk_entry_buffer_new (NULL,-1);
	get_Html_From_Website(D_Website_Select_Field_Buffer,D_Html_Field_Buffer); 

	// Html Field
	D_Html_Field = gtk_entry_new_with_buffer(D_Html_Field_Buffer);
	gtk_grid_attach (GTK_GRID (D_grid), D_Html_Field, 0, 2, 1, 1);

	gtk_widget_show_all (D_window);

}

void get_Html_From_Website(GtkEntryBuffer * url_buffer, GtkEntryBuffer *output_buffer) { 

    const char * val = gtk_entry_buffer_get_text( (GtkEntryBuffer *)  url_buffer );
    std::cout << "Value: " << val << "\n"; 

    std::string command = "wget -xi " + std::string(val); 

    std::cout << command << "\n"; 

    system(command.c_str()); 
    	
}

static void display_Html(GtkWidget *widget, gpointer *data) { 

}

//****************************************************************************************************************
// create_experiment_report
//****************************************************************************************************************

void on_window_destroy (GtkWidget *widget, gpointer data) {
	gtk_main_quit ();
}

/* Callback for close button */
void on_button_clicked (GtkWidget *button, GtkTextBuffer *buffer) {
	GtkTextIter start;
	GtkTextIter end;

	gchar *text;

	/* Obtain iters for the start and end of points of the buffer */
	gtk_text_buffer_get_start_iter (buffer, &start);
	gtk_text_buffer_get_end_iter (buffer, &end);

	/* Get the entire buffer text. */
	text = gtk_text_buffer_get_text (buffer, &start, &end, FALSE);

	/* Print the text */
	g_print ("%s", text);

	g_free (text);

	gtk_main_quit ();
}


static void create_experiment_report(GtkWidget *widget, gpointer *data) {

	GtkWidget *D_window;
	GtkWidget *D_grid;

//======================================================================================================

	GtkWidget * D_date_time_lable; 
	GtkWidget * D_date_time; 

//======================================================================================================

	D_update_time(); 

	D_videoPath += current_Time + std::string("/"); 
	D_audioPath += current_Time + std::string("/");
	D_ImagePath += current_Time + std::string("/");
	D_ImageListPath += current_Time + std::string("/");

	if (boost::filesystem::exists(D_videoPath)) {

		std::cout << "Weird Error, Dictory already exists though it shouldn't.\n"; 

	} else { 

		std::cout << "Creating Directory: " << D_videoPath << "\n"; 
//		boost::filesystem::create_directory(picDir); 

	}

//======================================================================================================	



//======================================================================================================

	GtkWidget *D_enter_topic_lable; 
	GtkEntryBuffer *D_Report_Topic_Buffer = gtk_entry_buffer_new (NULL,-1);

//======================================================================================================

	GtkWidget *D_Scrolled_Proceedure_Cont; 
	GtkWidget *D_enter_proceedure_lable; 
	GtkTextBuffer *D_Report_Procedure_Buffer = gtk_text_buffer_new (NULL);

	GtkWidget *D_Scrolled_Hypothesis_Cont; 
	GtkWidget *D_enter_hypothesis_lable; 
	GtkTextBuffer *D_Report_hypothesis_Buffer = gtk_text_buffer_new (NULL);

	GtkWidget *D_Scrolled_Results_Cont; 
	GtkWidget *D_enter_report_results_lable; 
	GtkTextBuffer *D_Report_Results_Buffer = gtk_text_buffer_new (NULL);

	GtkWidget *D_Scrolled_Analysis_Cont; 
	GtkWidget *D_enter_analysis_lable; 
	GtkTextBuffer *D_Report_Analysis_Buffer = gtk_text_buffer_new (NULL);

//======================================================================================================

	GtkWidget *D_open_camera_button; 
	GtkWidget *D_save_report_button; 
	GtkWidget *D_open_audacity_button; 	
	GtkWidget *D_record_video_button; 	

//======================================================================================================

	// Window Setup
    D_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(D_window), "DCAD");

    // Grid Setup
	D_grid = gtk_grid_new ();
	gtk_container_add (GTK_CONTAINER (D_window), D_grid);

//======================================================================================================

	D_date_time_lable = gtk_label_new ( std::string("Date/Time: ").c_str() );
	gtk_widget_set_hexpand (D_date_time_lable, FALSE);
	gtk_widget_set_vexpand (D_date_time_lable, FALSE);	
	gtk_widget_set_halign (D_date_time_lable, GTK_ALIGN_CENTER);
	gtk_grid_attach (GTK_GRID (D_grid), D_date_time_lable, 0, 0, 1, 1);

	D_date_time = gtk_label_new ( current_Time.c_str() );
	gtk_widget_set_hexpand (D_date_time, TRUE);
	gtk_widget_set_vexpand (D_date_time, FALSE);	
	gtk_widget_set_halign (D_date_time, GTK_ALIGN_CENTER);
	gtk_grid_attach (GTK_GRID (D_grid), D_date_time, 1, 0, 1, 1);

//======================================================================================================

	D_enter_topic_lable = gtk_label_new ("Topic:");
	gtk_widget_set_hexpand (D_enter_topic_lable, FALSE);
	gtk_widget_set_vexpand (D_enter_topic_lable, FALSE);	
	gtk_widget_set_halign (D_enter_topic_lable, GTK_ALIGN_START);
	gtk_grid_attach (GTK_GRID (D_grid), D_enter_topic_lable, 0, 1, 1, 1);

	D_Report_Topic = gtk_entry_new_with_buffer(D_Report_Topic_Buffer); 
	gtk_widget_set_hexpand (D_Report_Topic, TRUE);
	gtk_widget_set_vexpand (D_Report_Topic, FALSE);	
	gtk_widget_set_halign (D_Report_Topic, GTK_ALIGN_FILL);
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_Report_Topic, D_enter_topic_lable, GTK_POS_RIGHT, 1, 1);

//======================================================================================================

	D_enter_hypothesis_lable = gtk_label_new ("Hypothesis:");
	gtk_widget_set_hexpand (D_enter_hypothesis_lable, FALSE);
	gtk_widget_set_vexpand (D_enter_hypothesis_lable, FALSE);	
	gtk_widget_set_halign (D_enter_hypothesis_lable, GTK_ALIGN_START);
	gtk_widget_set_valign (D_enter_hypothesis_lable, GTK_ALIGN_START);	
//	g_signal_connect (D_enter_hypothesis_lable, "clicked", G_CALLBACK (), NULL);			
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_enter_hypothesis_lable, D_enter_topic_lable, GTK_POS_BOTTOM, 1, 1);

	D_Report_hypothesis = gtk_text_view_new_with_buffer(D_Report_hypothesis_Buffer); 
	gtk_widget_set_hexpand (D_Report_hypothesis, TRUE);
	gtk_widget_set_vexpand (D_Report_hypothesis, TRUE);	
	gtk_widget_set_halign (D_Report_hypothesis, GTK_ALIGN_FILL);
	gtk_widget_set_valign (D_Report_hypothesis, GTK_ALIGN_FILL);
//	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_Report_hypothesis, D_enter_hypothesis_lable, GTK_POS_RIGHT, 1, 1);

	D_Scrolled_Hypothesis_Cont = gtk_scrolled_window_new (NULL, NULL);
	gtk_container_add (GTK_CONTAINER (D_Scrolled_Hypothesis_Cont), D_Report_hypothesis);
	gtk_widget_set_hexpand (D_Scrolled_Hypothesis_Cont, TRUE);
	gtk_widget_set_vexpand (D_Scrolled_Hypothesis_Cont, TRUE);	
	gtk_widget_set_halign (D_Scrolled_Hypothesis_Cont, GTK_ALIGN_FILL);
	gtk_widget_set_valign (D_Scrolled_Hypothesis_Cont, GTK_ALIGN_FILL);
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_Scrolled_Hypothesis_Cont, D_enter_hypothesis_lable, GTK_POS_RIGHT, 1, 1);

//======================================================================================================

	D_enter_proceedure_lable = gtk_label_new ("Proceedure:");
	gtk_widget_set_hexpand (D_enter_proceedure_lable, FALSE);
	gtk_widget_set_vexpand (D_enter_proceedure_lable, FALSE);	
	gtk_widget_set_halign (D_enter_proceedure_lable, GTK_ALIGN_START);
	gtk_widget_set_valign (D_enter_proceedure_lable, GTK_ALIGN_START);	
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_enter_proceedure_lable, D_enter_hypothesis_lable, GTK_POS_BOTTOM, 1, 1);

	D_Report_Procedure = gtk_text_view_new_with_buffer(D_Report_Procedure_Buffer); 
	gtk_widget_set_hexpand (D_Report_Procedure, TRUE);
	gtk_widget_set_vexpand (D_Report_Procedure, TRUE);	
	gtk_widget_set_halign (D_Report_Procedure, GTK_ALIGN_FILL);
	gtk_widget_set_valign (D_Report_Procedure, GTK_ALIGN_FILL);
//	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_Report_Procedure, D_enter_proceedure_lable, GTK_POS_RIGHT, 1, 1);

	D_Scrolled_Proceedure_Cont = gtk_scrolled_window_new (NULL, NULL);
	gtk_container_add (GTK_CONTAINER (D_Scrolled_Proceedure_Cont), D_Report_Procedure);
	gtk_widget_set_hexpand (D_Scrolled_Proceedure_Cont, TRUE);
	gtk_widget_set_vexpand (D_Scrolled_Proceedure_Cont, TRUE);	
	gtk_widget_set_halign (D_Scrolled_Proceedure_Cont, GTK_ALIGN_FILL);
	gtk_widget_set_valign (D_Scrolled_Proceedure_Cont, GTK_ALIGN_FILL);
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_Scrolled_Proceedure_Cont, D_enter_proceedure_lable, GTK_POS_RIGHT, 1, 1);

//======================================================================================================

	D_enter_report_results_lable = gtk_label_new ("Results:");
	gtk_widget_set_hexpand (D_enter_report_results_lable, FALSE);
	gtk_widget_set_vexpand (D_enter_report_results_lable, FALSE);	
	gtk_widget_set_halign (D_enter_report_results_lable, GTK_ALIGN_START);
	gtk_widget_set_valign (D_enter_report_results_lable, GTK_ALIGN_START);	
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_enter_report_results_lable, D_enter_proceedure_lable, GTK_POS_BOTTOM, 1, 1);

	D_Report_Results = gtk_text_view_new_with_buffer(D_Report_Results_Buffer); 
	gtk_widget_set_hexpand (D_Report_Results, TRUE);
	gtk_widget_set_vexpand (D_Report_Results, TRUE);	
	gtk_widget_set_halign (D_Report_Results, GTK_ALIGN_FILL);
	gtk_widget_set_valign (D_Report_Results, GTK_ALIGN_FILL);
//	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_Report_Results, D_enter_report_results_lable, GTK_POS_RIGHT, 1, 1);

	D_Scrolled_Results_Cont = gtk_scrolled_window_new (NULL, NULL);
	gtk_container_add (GTK_CONTAINER (D_Scrolled_Results_Cont), D_Report_Results);
	gtk_widget_set_hexpand (D_Scrolled_Results_Cont, TRUE);
	gtk_widget_set_vexpand (D_Scrolled_Results_Cont, TRUE);	
	gtk_widget_set_halign (D_Scrolled_Results_Cont, GTK_ALIGN_FILL);
	gtk_widget_set_valign (D_Scrolled_Results_Cont, GTK_ALIGN_FILL);
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_Scrolled_Results_Cont, D_enter_report_results_lable, GTK_POS_RIGHT, 1, 1);

//======================================================================================================

	D_enter_analysis_lable = gtk_label_new ("Analysis:");
	gtk_widget_set_hexpand (D_enter_analysis_lable, FALSE);
	gtk_widget_set_vexpand (D_enter_analysis_lable, FALSE);	
	gtk_widget_set_halign (D_enter_analysis_lable, GTK_ALIGN_START);
	gtk_widget_set_valign (D_enter_analysis_lable, GTK_ALIGN_START);	
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_enter_analysis_lable, D_enter_report_results_lable, GTK_POS_BOTTOM, 1, 1);

	D_Report_Analysis = gtk_text_view_new_with_buffer(D_Report_Analysis_Buffer); 
	gtk_widget_set_hexpand (D_Report_Analysis, TRUE);
	gtk_widget_set_vexpand (D_Report_Analysis, TRUE);	
	gtk_widget_set_halign (D_Report_Analysis, GTK_ALIGN_FILL);
	gtk_widget_set_valign (D_Report_Analysis, GTK_ALIGN_FILL);
//	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_Report_Analysis, D_enter_analysis_lable, GTK_POS_RIGHT, 1, 1);

	D_Scrolled_Analysis_Cont = gtk_scrolled_window_new (NULL, NULL);
	gtk_container_add (GTK_CONTAINER (D_Scrolled_Analysis_Cont), D_Report_Analysis);
	gtk_widget_set_hexpand (D_Scrolled_Analysis_Cont, TRUE);
	gtk_widget_set_vexpand (D_Scrolled_Analysis_Cont, TRUE);	
	gtk_widget_set_halign (D_Scrolled_Analysis_Cont, GTK_ALIGN_FILL);
	gtk_widget_set_valign (D_Scrolled_Analysis_Cont, GTK_ALIGN_FILL);
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_Scrolled_Analysis_Cont, D_enter_analysis_lable, GTK_POS_RIGHT, 1, 1);

//======================================================================================================

    // Open Image Constructor Button Code
	D_open_camera_button = gtk_button_new_with_label ("Open Camera");
	g_signal_connect (D_open_camera_button, "clicked", G_CALLBACK (open_camera), NULL);
//	g_signal_connect_swapped (D_open_image_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_open_camera_button, D_enter_analysis_lable, GTK_POS_BOTTOM, 1, 1);

//======================================================================================================

    // Open Image Constructor Button Code
	D_save_report_button = gtk_button_new_with_label ("Save Report");
	g_signal_connect (D_save_report_button, "clicked", G_CALLBACK (NULL), NULL);
//	g_signal_connect_swapped (D_open_image_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_save_report_button, D_open_camera_button, GTK_POS_BOTTOM, 1, 1);

//======================================================================================================

    // Open Audacity Constructor Button Code
	D_open_audacity_button = gtk_button_new_with_label ("Open Audacity");
	g_signal_connect (D_open_audacity_button, "clicked", G_CALLBACK (D_Open_Audacity), NULL);
//	g_signal_connect_swapped (D_open_image_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_open_audacity_button, D_save_report_button, GTK_POS_BOTTOM, 1, 1);

//======================================================================================================

    // Open Audacity Constructor Button Code
	D_record_video_button = gtk_button_new_with_label ("Record Video");
	g_signal_connect (D_record_video_button, "clicked", G_CALLBACK (D_record_Video), NULL);
//	g_signal_connect_swapped (D_open_image_button, "clicked", G_CALLBACK (gtk_widget_destroy), D_window);
	gtk_grid_attach_next_to (GTK_GRID (D_grid), D_record_video_button, D_open_audacity_button, GTK_POS_BOTTOM, 1, 1);

//======================================================================================================

	gtk_widget_show_all (D_window);

}

//****************************************************************************************************************
// Extra Functions
//****************************************************************************************************************

std::string slistFilesInDirectory(std::string pathToDirectory) { 

	boost::filesystem::path p (pathToDirectory);  
	std::string output = ""; 

	if (boost::filesystem::exists(p)) {
		if (boost::filesystem::is_regular_file(p)) { 
			std::cout << p << " size is " << boost::filesystem::file_size(p) << '\n';
		} else if (boost::filesystem::is_directory(p)) {
			std::cout << p << " is a directory\n";		
			boost::filesystem::directory_iterator end_itr; 
			for ( boost::filesystem::directory_iterator itr( p ); itr != end_itr; ++itr ) {
				std::cout << itr->path().filename() << "\n"; 
				output += itr->path().filename().string() + (std::string)"\n"; 
			}

		} else {
			std::cout << p << " exists, but is neither a regular file nor a directory\n";		
		}
	} else {	
	 	std::cout << p << " does not exist\n";		
	}

	std::cout << "output:\n" << output << "\n"; 

	return output; 

}

void vlistFilesInDirectory(std::string pathToDirectory, std::vector<std::string> &output) { 

	boost::filesystem::path p (pathToDirectory);  

	if (boost::filesystem::exists(p)) {
		if (boost::filesystem::is_regular_file(p)) { 
			std::cout << p << " size is " << boost::filesystem::file_size(p) << '\n';
		} else if (boost::filesystem::is_directory(p)) {
			std::cout << p << " is a directory\n";		
			boost::filesystem::directory_iterator end_itr; 
			for ( boost::filesystem::directory_iterator itr( p ); itr != end_itr; ++itr ) {
				std::cout << itr->path().filename() << "\n"; 
				output.push_back(itr->path().filename().string()); 
			}

		} else {
			std::cout << p << " exists, but is neither a regular file nor a directory\n";		
		}
	} else {	
	 	std::cout << p << " does not exist\n";		
	}

}

void ShowMessage(const char* msg, const char* title) {

    std::cerr << title << std::endl;
    std::cerr << msg << std::endl;

    GtkWidget* dialog = gtk_message_dialog_new (NULL, GTK_DIALOG_MODAL, GTK_MESSAGE_INFO, GTK_BUTTONS_OK, title);
    gtk_message_dialog_format_secondary_text (GTK_MESSAGE_DIALOG (dialog), "%s", msg);
    gtk_dialog_run(GTK_DIALOG (dialog));
    gtk_widget_destroy(GTK_WIDGET(dialog));

}

void default_open_file_function(char * filename) { 
	std::cout << "Default File Processing\n"; 
	std::cout << "Selected file: " << filename << "\n"; 
}

void FileChooser(GtkWindow* window, void (*open_file_function)(char * filename) ) { 

	GtkWidget *dialog;
	GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
	gint res;

	dialog = gtk_file_chooser_dialog_new ("Open File",
	                                      window,
	                                      action,
	                                      ("_Cancel"),
	                                      GTK_RESPONSE_CANCEL,
	                                      ("_Open"),
	                                      GTK_RESPONSE_ACCEPT,
	                                      NULL);

	res = gtk_dialog_run (GTK_DIALOG (dialog));
	if (res == GTK_RESPONSE_ACCEPT) {
		char *filename;
		GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
		filename = gtk_file_chooser_get_filename(chooser);
		open_file_function(filename);
		g_free(filename);
	}

	gtk_widget_destroy (dialog);

}

void FileSelector(GtkWindow* window, std::string &fileSelected ) { 

	GtkWidget *dialog;
	GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
	gint res;

	dialog = gtk_file_chooser_dialog_new ("Open File",
	                                      window,
	                                      action,
	                                      ("_Cancel"),
	                                      GTK_RESPONSE_CANCEL,
	                                      ("_Open"),
	                                      GTK_RESPONSE_ACCEPT,
	                                      NULL);

	res = gtk_dialog_run (GTK_DIALOG (dialog));
	if (res == GTK_RESPONSE_ACCEPT) {

		GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
		fileSelected = std::string(gtk_file_chooser_get_filename(chooser));

	} else { 
		std::cout << "Error with GTK RESPONSE: " << res << "\n"; 
	}

	gtk_widget_destroy (dialog);

}

void read_Combo(GtkWidget * combo) { 
    const char * val = gtk_combo_box_text_get_active_text ((GtkComboBoxText*)combo);
    std::cout << "Selected: " << val << "\n"; 
}

void read_Field(GtkWidget *widget, gpointer *data) { 
    const char * val = gtk_entry_buffer_get_text( (GtkEntryBuffer *) data );
    std::cout << "Value: " << val << "\n"; 
} 

void add_View(GtkWidget * D_View, GtkTextBuffer * D_Buffer, GtkWidget * D_Grid, char * D_Text, int row, int col) { 

	// Input View
	D_View = gtk_text_view_new ();
	D_Buffer = gtk_text_view_get_buffer (GTK_TEXT_VIEW (D_View));
	gtk_text_buffer_set_text (D_Buffer, D_Text, -1);
	gtk_grid_attach (GTK_GRID (D_Grid), D_View, row, col, 1, 1);

}

void import_Minst_Dataset(int numberPerCategory) { 

	Neural_Net->setCategoryCount(10); 
	Neural_Net->setImagePerCategoryCount(numberPerCategory); 
	Neural_Net->setInformationAboutInputSpace(true,false); 
	Neural_Net->setInformationAboutOutputSpace(true); 

	std::ifstream file;

	std::string trainingListPath = "/home/drake/Documents/Code/DSOFT/mnistTrainingSets/trainingList"; 
	std::string trainingDataPath = "/home/drake/Documents/Code/DSOFT/mnistTrainingSets/mnist_png/training/"; 
	std::string trainingLabelPath = "/home/drake/Documents/Code/DSOFT/mnistTrainingSets/mnist_png/lables/"; 
	std::string tmp1; 
	std::string tmp2; 
	std::string line; 
	int count = 0; 

	for(int i = 0; i < 10; i++) { 

		tmp1 = trainingListPath  + std::to_string(i); 
		tmp2 = trainingLabelPath + std::to_string(i); 
		#ifdef __GENERAL_DEBUG__
			std::cout << tmp1 << "\n"; 
		#endif

		file.open(tmp1); 
		if (file.is_open()) {
			while ( getline (file,line) && count < numberPerCategory) {
				std::cout << line << "\n";
				tmp1 = trainingDataPath + std::to_string(i) + "/" + line; 
				Neural_Net->importTrainingInputData(tmp1); 
				Neural_Net->importTrainingLabelData(tmp2); 
				count++; 
			}
			count = 0; 
			file.close();
		} else { 
			std::cout << "File would not open\n"; 
		}

	}

}

void write_text_buffer_vectors_to_memory(std::vector<GtkTextBuffer*> input, std::string path ) { 
	std::cout << "Path: " << path << "\n"; 
}

static void D_Open_Audacity(GtkWidget *widget, gpointer data) { 
	system("audacity &"); 
}

static void D_record_Video(GtkWidget *widget, gpointer path) { 

    cv::Mat frame;

    std::vector<cv::Mat> capturedVideo; 

	std::cout << "Enter Camera to open: "; 
	int cam; 
	std::cin >> cam; 

	cv::VideoCapture stream1(cam);

    if (!stream1.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
    } else { 
	    std::cout << "Start grabbing" << std::endl
	        	  << "Press any key to terminate" << std::endl;
	    for (;;) {

	        stream1.read(frame);

	        capturedVideo.push_back(frame); 

	        if (frame.empty()) {
	            std::cerr << "ERROR! blank frame grabbed\n";
	            break;
	        }

	        cv::imshow("Live", frame);
	        if (cv::waitKey(5) >= 0) { 
	        	cv::destroyAllWindows(); 
	            break;
	        }
	    }

		std::cout << "Would you like to review the video captured? (1 - yes, else - no)"; 
		int review; 
		std::cin >> review; 

		if(review == 1) { 
			D_play_recorded_video(capturedVideo); 
		} else { 

		}

		std::cout << "Would you like to save the video captured? (1 - yes, else - no)"; 
		int save; 
		std::cin >> save;

		if(save == 1) { 
			D_save_openCV_Video(capturedVideo); 
		} else { 

		}		 

    }

}


void D_save_openCV_Video(std::vector<cv::Mat> video) {
//	std::cout << "Path: " << path << "\n"; 
} 

void D_play_recorded_video(std::vector<cv::Mat> video) { 
	int restart = 1; 
	while(restart == 1) { 
		for(auto it = video.begin(); it != video.end(); ++it) { 
			cv::imshow("Recorded Video", *it);
		}
		std::cout << "Would you like to restart the video? (1 - yes, else no)\n"; 
		std::cin >> restart; 
	}
	cv::destroyAllWindows(); 
}

void D_update_time() { 
	time_t now = time(0);

	// convert now to string form
	current_Time = std::string(ctime(&now));

	current_Time.erase(remove(current_Time.begin(), current_Time.end(), ' '), current_Time.end()); 
	current_Time.erase(remove(current_Time.begin(), current_Time.end(), ':'), current_Time.end()); 	
	current_Time.erase(remove(current_Time.begin(), current_Time.end(), '\n'), current_Time.end()); 
}