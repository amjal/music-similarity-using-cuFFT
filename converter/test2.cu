#include<omp.h>
#include<stdio.h>
void f(){
	for(int i =0; i<1000000; i ++);
}
int main(){
	int t1work = 0;
	int t2work = 0;
	int t3work = 0;
#pragma omp parallel num_threads(3)
	{
#pragma omp sections
		{
#pragma omp section
			{
				for(; t1work<100; t1work++){
					f();
					printf("t1 did %d\n", t1work);
				}
			}
#pragma omp section
			{
				for(; t2work<100; t2work++){
					while(t2work >= t1work);
					f();
					printf("t2 did %d\n", t2work);
				}
			}
#pragma omp section
			{
				for(; t3work<100; t3work++){
					while(t3work>=t2work);
					f();
					printf("t3 did %d\n", t3work);
				}
			}
		}
	}
	return 0;
}

