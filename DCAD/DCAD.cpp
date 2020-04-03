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

#ifndef __DCAD__
	#include "DCAD.h"
#endif

#ifndef __DEFINES__
	#include "../defines.h"
#endif

void setProjection(int w1, int h1); 
void changeSize(int w1,int h1); 
void renderBitmapString( float x, float y, float z, void *font, char *string); 
void restorePerspectiveProjection(); 
void computePos(float deltaMove);
void renderScene2();
void renderScene();
void renderScenesw1();
void renderSceneAll();
void processNormalKeys(unsigned char key, int xx, int yy);
void pressKey(int key, int xx, int yy);
void releaseKey(int key, int x, int y);
void mouseMove(int x, int y);
void mouseButton(int button, int state, int x, int y);
void init();
void mainLoop(); 
void setOrthographicProjection(); 
void processNormalKeys(unsigned char key, int xx, int yy);
void pressKey(int key, int xx, int yy);
void releaseKey(int key, int x, int y);
void mouseMove(int x, int y);
void mouseButton(int button, int state, int x, int y);

double rotate_y = 0; 
double rotate_x = 0;



// angle of rotation for the camera direction
float angle = 0.0f;

// actual vector representing the camera's direction
float lx=0.0f,lz=-1.0f, ly = 0.0f;

// XZ position of the camera
float x=0.0f, z=5.0f, y = 1.75f;

// the key states. These variables will be zero
//when no key is being presses
float deltaAngle = 0.0f;
float deltaMove  = 0;
int   xOrigin    = -1;

// width and height of the window
int h,w;

// variables to compute frames per second
int  frame;
long time_x, time_xbase;
char s[50];

// variables to hold window identifiers
int mainWindow, subWindow1;

int opengl_window_width = 900;
int opengl_window_height = 600;

//border between subwindows
int border = 6;

DCAD:: DCAD() { 

}

DCAD::~DCAD() { 

}

void mainLoop() { 

	// OpenGl Code

	// init GLUT and create main window
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(800,800);
	mainWindow = glutCreateWindow("3D Cad");

	// callbacks for main window
	glutDisplayFunc(renderSceneAll);
	glutReshapeFunc(changeSize);

	// Removing the idle function to save CPU and GPU
	//glutIdleFunc(renderSceneAll);
	init();

	// sub windows
	subWindow1 = glutCreateSubWindow(mainWindow, border,border,w-2*border, h/2 - border*3/2);
	glutDisplayFunc(renderScenesw1);
	init();

	// enter GLUT event processing cycle
	glutMainLoop();

}

void setProjection(int w1, int h1) {

	float ratio;
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	ratio = 1.0f * w1 / h1;
	// Reset the coordinate system before modifying
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Set the viewport to be the entire window
    glViewport(0, 0, w1, h1);

	// Set the clipping volume
	gluPerspective(45,ratio,0.1,1000);
	glMatrixMode(GL_MODELVIEW);

}

void changeSize(int w1,int h1) {

	if(h1 == 0) {
		h1 = 1;
	}

	// we're keeping these values cause we'll need them latter
	w = w1;
	h = h1;

	// set subwindow 1 as the active window
	glutSetWindow(subWindow1);
	// resize and reposition the sub window
	glutPositionWindow(border,border);
	glutReshapeWindow(w-border*3/2, h - border*3/2);
	setProjection(w-border*3/2, h - border*3/2);

}

void renderBitmapString( float x, float y, float z, void *font, char *string) {

	char *c;
	glRasterPos3f(x, y,z);
	for (c=string; *c != '\0'; c++) {
		glutBitmapCharacter(font, *c);
	}

}

void restorePerspectiveProjection() {

	glMatrixMode(GL_PROJECTION);
	// restore previous projection matrix
	glPopMatrix();

	// get back to modelview mode
	glMatrixMode(GL_MODELVIEW);

}

void setOrthographicProjection() {

	// switch to projection mode
	glMatrixMode(GL_PROJECTION);

	// save previous matrix which contains the
	//settings for the perspective projection
	glPushMatrix();

	// reset matrix
	glLoadIdentity();

	// set a 2D orthographic projection
	gluOrtho2D(0, w, h, 0);

	// switch back to modelview mode
	glMatrixMode(GL_MODELVIEW);

}

void computePos(float deltaMove) {

	x += deltaMove * lx * 0.1f;
	z += deltaMove * lz * 0.1f;

}

// Common Render Items for all subwindows
void renderScene2() {

	glColor3f(0.9f, 0.9f, 0.9f);
	glBegin(GL_QUADS);
		glVertex3f(-100.0f, 0.0f, -100.0f);
		glVertex3f(-100.0f, 0.0f,  100.0f);
		glVertex3f( 100.0f, 0.0f,  100.0f);
		glVertex3f( 100.0f, 0.0f, -100.0f);
	glEnd();

}

// Display func for main window
void renderScene() {

	glutSetWindow(mainWindow);
	glClear(GL_COLOR_BUFFER_BIT);
	glutSwapBuffers();

}

// Display func for sub window 1
void renderScenesw1() {

	glutSetWindow(subWindow1);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();
	gluLookAt(x, y, z,
		  x + lx,y + ly,z + lz,
		  0.0f,1.0f,0.0f);

	renderScene2();

	// display fps in the top window
 	frame++;

	time_x=glutGet(GLUT_ELAPSED_TIME);
	if (time_x - time_xbase > 1000) {
		sprintf(s,"Lighthouse3D - FPS:%4.2f",
			frame*1000.0/(time_x-time_xbase));
		time_xbase = time_x;
		frame = 0;
	}

	setOrthographicProjection();

	glPushMatrix();
	glLoadIdentity();
	renderBitmapString(5,30,0,GLUT_BITMAP_HELVETICA_12,s);
	glPopMatrix();

	restorePerspectiveProjection();

	glutSwapBuffers();

}

// Global render func
void renderSceneAll() {

	// check for keyboard movement
	if (deltaMove) {
		computePos(deltaMove);
		glutSetWindow(mainWindow);
		glutPostRedisplay();
	}

	renderScene();
	renderScenesw1();

}

// KEYBOARD 
void processNormalKeys(unsigned char key, int xx, int yy) {

	switch (key) {
		case 27 : glutDestroyWindow(mainWindow); exit(0); break; 
//		case 27 : glutDestroyWindow(mainWindow); exit(0); break; 		
	}

}

void pressKey(int key, int xx, int yy) {

	switch (key) {
		case GLUT_KEY_UP : deltaMove = 0.5f; break;
		case GLUT_KEY_DOWN : deltaMove = -0.5f; break;
	}
	glutSetWindow(mainWindow);
	glutPostRedisplay();

}

void releaseKey(int key, int x, int y) {

	switch (key) {
		case GLUT_KEY_UP :
		case GLUT_KEY_DOWN : deltaMove = 0;break;
	}

}

// -----------------------------------
//             MOUSE
// -----------------------------------
void mouseMove(int x, int y) {

	// this will only be true when the left button is down
	if (xOrigin >= 0) {

		// update deltaAngle
		deltaAngle = (x - xOrigin) * 0.001f;

		// update camera's direction
		lx = sin(angle + deltaAngle);
		lz = -cos(angle + deltaAngle);

		glutSetWindow(mainWindow);
		glutPostRedisplay();

	}

}

void mouseButton(int button, int state, int x, int y) {

	// only start motion if the left button is pressed
	if (button == GLUT_LEFT_BUTTON) {

		// when the button is released
		if (state == GLUT_UP) {
			angle += deltaAngle;
			deltaAngle = 0.0f;
			xOrigin = -1;
		}
		else  {// state = GLUT_DOWN
			xOrigin = x;

		}
	}

}

// -----------------------------------
//             MAIN and INIT
// -----------------------------------
void init() {

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	// register callbacks
	glutIgnoreKeyRepeat(1);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(pressKey);
	glutSpecialUpFunc(releaseKey);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);

}

