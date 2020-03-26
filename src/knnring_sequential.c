#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cblas.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>

#include "../inc/knnring.h"



//Functions Declaration
void swap_int(int *x, int *y);
void swap(double *x, double *y);
int partition (double *arr,int *index, int l, int r);
void kthSmallest(double *arr,int *index, int l,int r, int k);
void calculateDistances(double *distance,double * X, double * Y, int n, int m, int d);
void quicksort(double * arr, int* index,int low, int high);





knnresult kNN(double * X, double * Y, int n, int m, int d, int k)
{
    struct timeval startwtime,endwtime;
    double p_time;
    gettimeofday(&startwtime,NULL);
    //the struct we return as result initiliazed
    knnresult result;
    result.m = m;
    result.k = k;
    result.nidx = NULL;
    result.ndist = NULL;


  	//Allocating memory for distance matrix
  	double * distance_matrix = (double*) malloc(n*m * sizeof(double));


 	  //call function that calculates eucledian distances
  	calculateDistances(distance_matrix,X,Y,n ,m ,d);

	   //in order to use our quickselect algorithm that we implemented in exercise_1
	   // we need to calculate the transpose of matrix since quickselect returns k smallest elements
	   //in the beginning of an arr[l..r]
  	 double * distance_matrix_transpose = (double *)malloc(m*n*sizeof(double));
  	 for(int i=0; i<n; i++)
  	 {
    	 for(int j=0; j<m; j++)
    	 {
      		distance_matrix_transpose[j*n + i] = distance_matrix[i*m + j];
      	}
      }



    //allocate memory for index matrix
  	int * index_matrix = (int*) malloc(n*m * sizeof(int));

	//initiliaze index matrix(reminder that distance matrix is m x n now to fit quickselect)
	for(int i=0; i<m; i++)
	{
    	for(int j=0; j<n; j++) 
    	{
      		index_matrix[i*n+j]=j; 
      	}
    }


	//move k smallest to k first columns
	for(int i=0 ; i< m ;i++)
	{
		kthSmallest(distance_matrix_transpose,index_matrix,i*n,(i+1)*n-1, k);
	}

	//create 2 matrices to save the mxk results so we can sort them before we return them to the result
	double * temp_ndist = (double *) malloc(m*k * sizeof(double));
	int * temp_nidx = (int *) malloc (m * k * sizeof(int));

	//get the k first columns and add them to the result struct
	for(int i=0; i < m ; i++)
	{
		for(int j =0 ; j<k ;j++)
		{
	  		temp_ndist[i*k+j]=distance_matrix_transpose[i*n+j];
	  		temp_nidx[i*k+j]=index_matrix[i*n+j];
		}
	}

	//sort matrices m xk with quicksort
	for(int i = 0 ; i<m; i++)
	{
      quicksort(temp_ndist , temp_nidx , i*k , (i+1)*k-1);
  	}

  	result.ndist = temp_ndist;
  	result.nidx = temp_nidx;


	 free(distance_matrix_transpose);
	 free(index_matrix);

    gettimeofday(&endwtime,NULL);
    p_time = (double)((endwtime.tv_usec-startwtime.tv_usec)/1.0e6+endwtime.tv_sec-startwtime.tv_sec);
    printf("Time %f\n",p_time);
	
	return result;
	}




//function that calculates eucledian distances
void calculateDistances(double *distance,double * X, double * Y, int n, int m, int d)
{
  	//from pdf, we need to calculate 
	//D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).')

	double alpha=-2.0, beta=0.0;
	int lda=d, ldb=d, ldc=m;
	double limit = 0.00000001;



    //calculate -2*X*Y with cblas
    //http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html#gaeda3cbd99c8fb834a60a6412878226e1
    //https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm

    cblas_dgemm(CblasRowMajor , CblasNoTrans , CblasTrans , n, m , d , alpha , X , lda , Y , ldb , beta, distance , ldc);

    //sumX stores sum(X.^2,2) 
    //sumY sotres sum(Y.^2,2) 
    double * sumX = (double *) malloc(n *sizeof(double));
    double * sumY = (double *) malloc(m *sizeof(double));;


    //calculate sum(X.^2,2)
    for(int i=0 ;i<n; i++)
    {
	  	for(int j=0; j<d ;j++)
	  	{
	    	sumX[i] += X[i*d+j] * X[i*d+j];
	  	}
	  	
	}

	//calculate sum(Y.^2,2)
	for(int i=0;i<m;i++)
	{
	  for(int j=0; j<d ;j++)
	  {
	    sumY[i] +=Y[i*d+j]*Y[i*d+j];
	  }

	}

	//calculate distance
	for(int i=0;i<n;i++)
	{
	  for(int j=0; j<m ;j++)
	  {
	    distance[i*m+j] += sumX[i] + sumY[j];
	  }

	}

	//sqrt distances
	//limit to avoid values that are very small
	for(int i=0; i<n*m ; i++)
	{
		if (distance[i] < limit)
		{
			distance[i]= 0;
		}
		else
		{
	    	distance[i]=sqrt(fabs(distance[i]));
		}
	}



}






//implemented from project 1 (vptree)
/* QUICKSELECT IMPLEMENTATION
	-KTHSMALLEST
    -SWAP
    -PARTITION
    
slighty modified version of c++ implementation found here:
https://www.geeksforgeeks.org/quickselect-algorithm/
https://en.wikipedia.org/wiki/Quickselect(pseudocode)

*/

// This function returns k'th smallest element in arr[l..r] using QuickSort.
void kthSmallest(double *arr,int *index, int l,int r, int k)
{

    // If k is smaller than number of  
    // elements in array 
    if (k > 0 && k <= r - l + 1) 
    {             
        int pivot = partition(arr,index, l, r); 
  
        // If position is same as k 
        if (pivot - l == k - 1) 
            return; 
  
        // If position is more, recur  
        // for left subarray 
        if (pivot - l > k - 1) 
            return kthSmallest(arr,index, l, pivot - 1, k); 
  
        // Else recur for right subarray 
        return kthSmallest(arr,index, pivot + 1, r, k - pivot + l - 1);    
    } 
    return;

}

// Standard partition process of QuickSort().
int partition (double *arr,int *index, int l, int r)
{
    double x = arr[r];
    int i = l;

    for (int j = l; j <= r - 1; j++)
    {
        if (arr[j] <= x)
        {
            swap(&arr[i], &arr[j]);
            swap_int(&index[i], &index[j]);
            i++;            
        }
    }
    swap(&arr[i], &arr[r]);
    swap_int(&index[i], &index[r]);
    return i;
}

// Swaps two elements(double)
void swap(double *x, double *y)
{
    double temp = *x;
    *x = *y;
    *y = temp;
}
//swap two elements(int) for index array
void swap_int(int *x, int *y)
{
    int temp = *x;
    *x = *y;
    *y = temp;
}


//https://www.geeksforgeeks.org/quick-sort/
void quicksort(double * arr, int* index,int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[pi] is now
           at right place */
        int pi = partition(arr,index, low, high);

        quicksort(arr,index, low, pi - 1);  // Before pi
        quicksort(arr,index, pi + 1, high); // After pi
    }
}


