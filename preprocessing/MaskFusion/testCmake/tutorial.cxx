// A simple program that computes the square root of a number
  #include <stdio.h>
  #include <stdlib.h>
  #include <math.h>
  #include "TutorialConfig.h"


  int main (int argc, char *argv[])
  {
    if (argc < 2)
    {
      fprintf(stdout, "%s Version %d.%d\n", 
                argv[0],
                Tutorial_VERSION_MAJOR,
                Tutorial_VERSION_MINOR);
      fprintf(stdout, "Uage: %s number\n", argv[0]);
      return 1;
    }
    double inputValue = atof(argv[1]);
    double outputValue = sqrt(inputValue);
    fprintf(stdout, "The square root of %g is %g\n",
              inputValue, outputValue);
    
    int* ptr=new int[10];
    ptr[0]=20;
    while(int i=0;i<10;i++){
        printf("Value %d \n", ptr[i]);
    }
    delete[] ptr;

    return 0;
  }