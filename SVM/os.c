#include<stdio.h>

int main()
{
	char fname[100] = "";
	int n[100];
	printf("Enter File name:");
	scanf("%s",fname);
	FILE *f = fopen(fname,"r");
	if(f==NULL)
	{
		print("Done");
		exit(1);
	}
	int num = 0;
	fscan(f,"%1d",&num);

	for(int i=0;i<num;i++)
	{
		fscanf(f,"%1d",&n[i]);

	}
	for(int i=0;i<num;i++)
	{
		printf("%d\n",n[i]);
	}

}