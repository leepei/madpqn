#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <mpi.h>
#include "linear.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void print_null(const char *s) {}

void exit_with_help()
{
	if(mpi_get_rank() != 0)
		mpi_exit(1);
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-P type :  set type of problem (default 0)\n"
	"	 0 -- L1-regularized logistic regression\n"
	"	 1 -- L1-regularized least-square regression (LASSO)\n"
	"	 2 -- L1-regularized L2-loss support vector classification\n"
	"	 3 -- GroupLASSO-regularized multinomial logistic regression\n"
	"-c cost : set the parameter C (default 1)\n"
	"-I iter : max inner iterations per round\n"
	"-a epsilon0 : adaptive stopping condition of the inner solver (default 0.01)\n"
	"-e epsilon1 : set tolerance of termination criterion (default 0.0001)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-q : quiet mode (no outputs)\n"
	"-m m : use information from the past m iterations (default 10)\n"
	"-p : permute features\n"
    "-S : disable smooth stage\n"
	);
	mpi_exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"[rank %d] Wrong input format at line %d\n", mpi_get_rank(), line_num);
	mpi_exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);

struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model* model_;
double bias;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int global_l, global_n;

	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);

	global_l = prob.l;
	global_n = prob.n;
	mpi_allreduce(&global_l, 1, MPI_INT, MPI_SUM);
	mpi_allreduce(&global_n, 1, MPI_INT, MPI_MAX);
	prob.n = global_n;
	prob.global_l = global_l;
	error_msg = check_parameter(&prob,&param);

	if(mpi_get_rank()==0)
		printf("#instance = %d, #feature = %d\n", global_l, global_n);
	if(error_msg)
	{
		if(mpi_get_rank()==0)
			fprintf(stderr,"ERROR: %s\n", error_msg);
		mpi_exit(1);
	}

	model_=train(&prob, &param);
	if(save_model(model_file_name, model_))
	{
		fprintf(stderr,"[rank %d] can't save model to file %s\n", mpi_get_rank(), model_file_name);
		mpi_exit(1);
	}
	free_and_destroy_model(&model_);
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

	MPI_Finalize();
	return 0;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.C = 1.0;
	param.eps = 0.0001; // see setting below
	param.m = 10;
	param.eta = 1e-4;
	param.max_inner_iter = 100;
	param.inner_eps = 0.01;
	param.permute_features = false;
	param.disable_smooth = false;
	param.problem_type = L1R_LR;
	
	bias = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'P':
				param.problem_type = atoi(argv[i]);
				break;

			case 'I':
				param.max_inner_iter = atoi(argv[i]);
				break;

			case 'a':
				param.inner_eps= atof(argv[i]);
				break;

			case 'c':
				param.C = atof(argv[i]);
				break;

			case 'm':
				param.m = atoi(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'q':
				print_func = &print_null;
				i--;
				break;

			case 'p':
				param.permute_features = true;
				i--;
				break;

			case 'S':
				param.disable_smooth = true;
				i--;
				break;


			default:
				if(mpi_get_rank() == 0)
					fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();
	int multiple = 1;
	if (mpi_get_size() == 1)
		multiple = 0;
	if (multiple)
	{
		char tmp_fmt[19];
		sprintf(tmp_fmt,"%%s.%%0%dd", int(log10(mpi_get_size()-1))+1);
		sprintf(input_file_name,tmp_fmt, argv[i], mpi_get_rank());
	}
	else
		strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}

	// default solver for parameter selection is L2R_L2LOSS_SVC
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"[rank %d] can't open input file %s\n",mpi_get_rank(),filename);
		mpi_exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);

	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

	fclose(fp);
}
