#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include <mpi.h>
#include <set>
#include <map>
#include <vector>
#include <algorithm>

#ifndef TIMER
#define TIMER
#define NS_PER_SEC 1000000000
double wall_time_diff(int64_t ed, int64_t st)
{
	return (double)(ed-st)/(double)NS_PER_SEC;
}

int64_t wall_clock_ns()
{
#if __unix__
	struct timespec tspec;
	clock_gettime(CLOCK_MONOTONIC, &tspec);
	return tspec.tv_sec*NS_PER_SEC + tspec.tv_nsec;
#else
#if __MACH__
	return 0;
#else
	struct timeval tv;
	gettimeofday( &tv, NULL );
	return tv.tv_sec*NS_PER_SEC + tv.tv_usec*1000;
#endif
#endif
}
#endif

double communication;
double global_n;

typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	if(mpi_get_rank()!=0)
		return;
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif
class sparse_operator
{
public:
	static double dot(const double *s, const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += s[x->index-1]*x->value;
			x++;
		}
		return (ret);
	}

	static void axpy(const double a, const feature_node *x, double *y)
	{
		while(x->index != -1)
		{
			y[x->index-1] += a*x->value;
			x++;
		}
	}
};

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);
extern int dgemv_(char *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
extern void dgetrf_(int *, int *, double *, int *, int *, int *);
extern void dgetri_(int *, double *, int *, int *, double *, int *, int *);

#ifdef __cplusplus
}
#endif
void inverse(double* A, int N) //compute the inverse of a symmetric PD matrix
{
	int *IPIV = new int[N+1];
	int LWORK = N*N;
	double *WORK = new double[LWORK];
	int INFO;

	dgetrf_(&N,&N,A,&N,IPIV,&INFO);
	dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

	delete[] IPIV;
	delete[] WORK;
}

class regularized_fun
{
public:
	virtual double fun(double *w) = 0;
	virtual double f_update(double *w, double *step, double Q, double eta, int *index = NULL, int index_length = 0, int localstart = 0, int locallength = 0) = 0;
	virtual void loss_grad(double *w, double *g, const std::vector<int> &fullindex = std::vector<int>{-1}) = 0;
	virtual void Hv(double *s, double *Hs, const std::vector<int> &fullindex = std::vector<int>{-1}, double *w = NULL) = 0;
	virtual double vHv(double *v, const std::vector<int> &fullindex = std::vector<int>{-1}) = 0; //For estimating the starting step size in sparsa
	virtual void get_diag_preconditioner(double *M, const std::vector<int> &fullindex = std::vector<int>{-1}, double *w = NULL) = 0;
	virtual void full_grad(double *w, double *g, std::vector<int> &index, double *full_g, std::vector<int> &local_nonzero_set) = 0;
	virtual void setselection(double *w, double *loss_g, std::vector<int> &prev_index, std::vector<int> &curr_index, double shrinkage = 1.0) = 0;
	virtual int get_nr_variable(void) = 0;
	virtual double armijo_line_search(double *step, double *w, double *loss_g, double *step_size, double eta, int *num_line_search_steps, double *delta_ret, int *index = NULL, int indexlength = 0, int localstart = 0, int locallength = 0) = 0;
	virtual double smooth_line_search(double *w, double *smooth_step, double delta, double eta, std::vector<int> &index, double *fnew) = 0;
	virtual void prox_grad(double *w, double *g, double *step, double *oldd, double alpha = 0, int index_length = -1) = 0;
	virtual double regularizer(double *w, int length) = 0;
	virtual void get_local_index_start_and_length(int global_length, int *local_start, int *local_length) = 0;

	int start; // Every node handles only a part of the w vector, from start to start + length - 1;
	int length;
	int *recv_count;
	int *displace;

protected:
	double C;
	double *z;
	double *Xw;
	double *D;
	const problem *prob;
	double reg;
	double current_f;
};

class l1r_fun: public regularized_fun
{
public:
	l1r_fun(const problem *prob, double C);
	virtual ~l1r_fun(){}

	double fun(double *w);
	double f_update(double *w, double *step, double Q, double eta, int *index = NULL, int index_length = 0, int localstart = 0, int locallength = 0);
	void loss_grad(double *w, double *g, const std::vector<int> &fullindex = std::vector<int>{-1});
	void Hv(double *s, double *Hs, const std::vector<int> &fullindex = std::vector<int>{-1}, double *w = NULL);
	double vHv(double *v, const std::vector<int> &fullindex = std::vector<int>{-1}); //For estimating the starting step size in sparsa
	void get_diag_preconditioner(double *M, const std::vector<int> &fullindex = std::vector<int>{-1}, double *w = NULL);
	void full_grad(double *w, double *g, std::vector<int> &index, double *full_g, std::vector<int> &local_nonzero_set);
	void setselection(double *w, double *loss_g, std::vector<int> &prev_index, std::vector<int> &curr_index, double shrinkage = 1.0);
	int get_nr_variable(void);
	double armijo_line_search(double *step, double *w, double *loss_g, double *step_size, double eta, int *num_line_search_steps, double *delta_ret, int *index = NULL, int indexlength = 0, int localstart = 0, int locallength = 0);
	double smooth_line_search(double *w, double *smooth_step, double delta, double eta, std::vector<int> &index, double *fnew);
	void Xv(double *v, double *Xv, const int *index = NULL, int index_length = 0);
	void prox_grad(double *w, double *g, double *step, double *oldd, double alpha = 0, int index_length = -1);
	double regularizer(double *w, int length);
	void get_local_index_start_and_length(int global_length, int *local_start, int *local_length);
	
protected:
	virtual double loss(int i, double wx_i) = 0;
	virtual void update_zD() = 0;
	void XTv(double *v, double *XTv, const std::vector<int> &fullindex = std::vector<int>{-1});
};

class l1r_lr_fun: public l1r_fun
{
public:
	l1r_lr_fun(const problem *prob, double C);
	~l1r_lr_fun();
	
protected:
	double loss(int i, double xw_i);
	void update_zD();
};

class lasso: public l1r_lr_fun
{
public:
	lasso(const problem *prob, double C);
	~lasso();

private:
	double loss(int i, double xw_i); // Loss is (w^T x - y)^2 / 2
	void update_zD();

};

class l1r_l2_svc_fun: public l1r_lr_fun
{
public:
	l1r_l2_svc_fun(const problem *prob, double C);
	~l1r_l2_svc_fun();

private:
	double loss(int i, double xw_i);
	void update_zD();

};

class grouplasso_mlr_fun: public regularized_fun
{
public:
	grouplasso_mlr_fun(const problem *prob, double C, int nr_class);
	~grouplasso_mlr_fun();

	double fun(double *w);
	double f_update(double *w, double *step, double Q, double eta, int *index = NULL, int index_length = 0, int localstart = 0, int locallength = 0);
	void loss_grad(double *w, double *g, const std::vector<int> &fullindex = std::vector<int>{-1});
	void Hv(double *s, double *Hs, const std::vector<int> &fullindex = std::vector<int>{-1}, double *w = NULL);
	double vHv(double *v, const std::vector<int> &fullindex = std::vector<int>{-1}); //For estimating the starting step size in sparsa
	void get_diag_preconditioner(double *M, const std::vector<int> &fullindex = std::vector<int>{-1}, double *w = NULL);
	void full_grad(double *w, double *g, std::vector<int> &index, double *full_g, std::vector<int> &local_nonzero_set);
	void setselection(double *w, double *loss_g, std::vector<int> &prev_index, std::vector<int> &curr_index, double shrinkage = 1.0);
	int get_nr_variable(void);
	double armijo_line_search(double *step, double *w, double *loss_g, double *step_size, double eta, int *num_line_search_steps, double *delta_ret, int *index = NULL, int indexlength = 0, int localstart = 0, int locallength = 0);
	double smooth_line_search(double *w, double *smooth_step, double delta, double eta, std::vector<int> &index, double *fnew);
	void prox_grad(double *w, double *g, double *step, double *oldd, double alpha = 0, int index_length = -1);
	double regularizer(double *w, int length);
	void get_local_index_start_and_length(int global_length, int *local_start, int *local_length);
	
protected:
	int nr_class;
	int full_len;
	int Xw_length;
	void Xv(double *v, double *Xv, const int *index = NULL, int index_length = 0);
	void XTv(double *v, double *XTv, const std::vector<int> &fullindex = std::vector<int>{-1});
	double loss(int i, double* wx_i);
	void update_zD();
	int localfeatures;
	int localfeature_start;
};

l1r_fun::l1r_fun(const problem *prob, double C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	Xw = new double[l];
	D = new double[l];
	this->C = C;
	int w_size = get_nr_variable();
	int nr_node = mpi_get_size();
	int rank = mpi_get_rank();
	int shift = int(ceil(double(w_size) / double(nr_node)));
	recv_count = new int[nr_node];
	displace = new int[nr_node];
	this->start = shift * rank;
	this->length = min(max(w_size - start, 0), shift);
	if (length == 0)
		start = 0;
	int counter = 0;
	for (int i=0;i<nr_node;i++)
	{
		recv_count[i] = shift;
		displace[i] = counter;
		counter += shift;
		if (counter >= w_size)
		{
			counter = 0;
			shift = 0;
		}
		else if (counter + shift > w_size)
			shift = w_size - counter;
	}
}

void l1r_fun::get_local_index_start_and_length(int global_length, int *local_start, int *local_length)
{
	int nr_node = mpi_get_size();
	int rank = mpi_get_rank();
	int shift = int(ceil(double(global_length) / double(nr_node)));
	int idx_start = shift * rank;
	int idx_length = min(max(global_length - idx_start, 0), shift);
	if (idx_length == 0)
		idx_start = 0;
	*local_start = idx_start;
	*local_length = idx_length;
}

double l1r_fun::regularizer(double *w, int reg_length)
{
	double ret = 0;
	for (int i=0;i<reg_length;i++)
		ret += fabs(w[i]);
	return ret;
}

double l1r_fun::fun(double *w)
{
	int i;
	double f=0;
	int l=prob->l;
	double w_size = get_nr_variable();
	int nnz = 0;
	for (i=0;i<w_size;i++)
		nnz += (w[i] != 0);
	if (nnz == 0)
		memset(Xw, 0, sizeof(double) * l);
	else if ((double)nnz / w_size < 0.5)
	{
		int *index = new int[nnz];
		double *subw = new double[nnz];
		int counter = 0;
		for (i=0;i<w_size;i++)
			if (w[i] != 0)
			{
				subw[counter] = w[i];
				index[counter++] = i;
			}

		Xv(subw, Xw, index, nnz);
		delete[] index;
		delete[] subw;
	}
	else
		Xv(w, Xw);
	reg = regularizer(w + start, length);
	for(i=0;i<l;i++)
		f += loss(i, Xw[i]);
	double buffer[2] = {f, reg};
	mpi_allreduce(buffer, 2, MPI_DOUBLE, MPI_SUM);
	communication += 2 / global_n;

	reg = buffer[1];
	f = C * buffer[0] + reg;

	current_f = f;
	return(f);
}

double l1r_fun::f_update(double *w, double *step, double Q, double eta, int *index, int index_length, int localstart, int locallength)
{
	if (Q > 0)
		return current_f;
	int *localindex = NULL;
	double *localstep;
	int i;
	int inc = 1;
	int l = prob->l;
	double one = 1.0;
	double *substep;
	int *subindex;
	int nnz = 0;
	int len = index_length;
	int w_size = get_nr_variable();
	if (len == 0)
		len = w_size;


	double reg_diff = 0;

	if (index != NULL) // If a subset is selected
	{
		localindex = index + localstart;
		localstep = step + localstart;
		for (i=0;i<locallength;i++)
			reg_diff += fabs(w[localindex[i]] + localstep[i]) - fabs(w[localindex[i]]);
	}
	else
	{
		double *localw = w + start;
		localstep = step + start;
		for (i=0;i<length;i++)
			reg_diff += fabs(localw[i] + localstep[i]) - fabs(localw[i]);
	}

	for (i=0;i<len;i++)
		nnz += (step[i] != 0);
	if (nnz == 0)
		memset(z, 0, sizeof(double) * l);
	else if (nnz < len * 0.5)
	{
		substep = new double[nnz];
		subindex = new int[nnz];
		int counter = 0;
		if (index == NULL)
		{
			for (i=0;i<w_size;i++)
				if (step[i] != 0)
				{
					substep[counter] = step[i];
					subindex[counter++] = i;
				}
		}
		else
		{
			for (i=0;i<len;i++)
				if (step[i] != 0)
				{
					substep[counter] = step[i];
					subindex[counter++] = index[i];
				}
		}
		Xv(substep, z, subindex, nnz);
		delete[] substep;
		delete[] subindex;
	}
	else
		Xv(step, z, index, index_length);


	daxpy_(&l, &one, z, &inc, Xw, &inc);
	double f_new = 0;
	for(i=0; i<l; i++)
		f_new += loss(i, Xw[i]);
	f_new *= C;
	f_new += reg_diff;
	//If profiling gets sparsity, should consider and only updating individual losses
	//This should be the case when data has sparsity and the update is extremely sparse
	mpi_allreduce(&f_new, 1, MPI_DOUBLE, MPI_SUM);
	communication += 1 / global_n;
	f_new += reg;
	if (f_new - current_f <= eta * Q)
	{
		mpi_allreduce(&reg_diff, 1, MPI_DOUBLE, MPI_SUM);
		communication += 1 / global_n;
		current_f = f_new;
		reg += reg_diff;
	}
	else
	{
		double factor = -1;
		daxpy_(&l, &factor, z, &inc, Xw, &inc);
	}

	return current_f;
}

void l1r_fun::loss_grad(double *w, double *g, const std::vector<int> &fullindex)
{
	int i;
	int n = prob->n;
	int global_index_length = (int)fullindex.size();

	update_zD();

	XTv(z, g, fullindex);
	if (fullindex[0] == -1)
	{
		mpi_allreduce(g, n, MPI_DOUBLE, MPI_SUM);
		communication += 1.0;
	}
	else
	{
		mpi_allreduce(g, global_index_length, MPI_DOUBLE, MPI_SUM);
		for (i = global_index_length - 1; i >= 0; i--)
		{
			if (fullindex[i] == i)
				break;
			g[fullindex[i]] = g[i];
			g[i] = 0;
		}
		communication += (double) global_index_length / global_n;
	}
}

void l1r_fun::Hv(double *s, double *Hs, const std::vector<int> &fullindex, double *w)
{
	int i;
	int l=prob->l;
	int w_size=(int)fullindex.size();

	Xv(s, z, &fullindex[0], w_size);
	for(i=0;i<l;i++)
		z[i] = C*D[i]*z[i];

	XTv(z, Hs, fullindex);
	mpi_allreduce(Hs, w_size, MPI_DOUBLE, MPI_SUM);
	communication += w_size / global_n;
}

double l1r_fun::vHv(double *s, const std::vector<int> &index)
{
	int i;
	int inc = 1;
	int l=prob->l;

	if (index[0] == -1)
		Xv(s, z);
	else
		Xv(s, z, index.data(), (int) index.size());
	double alpha = 0;
	for(i=0;i<l;i++)
		alpha += z[i] * z[i] * D[i];// v^THv = C (Xv)^T D (Xv)
	double buffer[2];
	double norm2;
	if (index[0] == -1)
		norm2 = ddot_(&length, s + start, &inc, s + start, &inc);
	else
	{
		int index_size = (int) index.size();
		int nr_node = mpi_get_size();
		int rank = mpi_get_rank();
		int shift = int(ceil(double(index_size) / double(nr_node)));
		int indexstart = shift * rank;
		int indexlength = min(max(index_size - indexstart, 0), shift);
		if (indexlength == 0)
			indexstart = 0;
		norm2 = ddot_(&indexlength, s + indexstart, &inc, s + indexstart, &inc);
	}
	buffer[0] = alpha;
	buffer[1] = norm2;
	mpi_allreduce(buffer, 2, MPI_DOUBLE, MPI_SUM);
	norm2 = buffer[1];
	alpha = buffer[0] * C / norm2;
	communication += 2 / global_n;

	return alpha;
}

void l1r_fun::get_diag_preconditioner(double *M, const std::vector<int> &fullindex, double *w)
{
	int i;
	int w_size=(int)fullindex.size();
	feature_node **x = prob->x;
	for (i=0; i<w_size; i++)
		M[i] = 0.0;

	for (i=0; i<w_size; i++)
	{
		feature_node *s;
		s=x[fullindex[i]];
		while (s->index!=-1)
		{
			M[i] += s->value*s->value*C*D[s->index - 1];
			s++;
		}
	}
	mpi_allreduce(M, w_size, MPI_DOUBLE, MPI_SUM);
	communication += w_size / global_n;
}

void l1r_fun::full_grad(double *w, double *g, std::vector<int> &index, double *full_g, std::vector<int> &nonzero_set)
{
	for (int i=0, full_i=0;i<(int) index.size();i++)
	{
		int j = index[i];
		if (w[j] != 0)
		{
			nonzero_set.push_back(j);
			full_g[full_i++] = g[j] + (2 * (w[j] > 0) - 1);
		}
	}
}

void l1r_fun::Xv(double *v, double *Xv, const int *index, int index_length)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	if (index_length > 0)
		w_size = index_length;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=0;
	for(i=0;i<w_size;i++)
		if (v[i] != 0)
		{
			feature_node *s;
			if (index_length > 0)
				s = x[index[i]];
			else
				s = x[i];
			sparse_operator::axpy(v[i], s, Xv);
		}
}

void l1r_fun::XTv(double *v, double *XTv, const std::vector<int> &fullindex)
{
	int i;
	int w_size=(int)fullindex.size();
	if (fullindex[0] == -1)
		w_size = get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
	{
		feature_node * s;
		if (fullindex[0] == -1)
			s=x[i];
		else
			s=x[fullindex[i]];
		XTv[i] = sparse_operator::dot(v, s);
	}
}


void l1r_fun::setselection(double *w, double *loss_g, std::vector<int> &prev_index, std::vector<int> &curr_index, double shrinkage) //Some preliminary set selection strategy
{
	// (input)  prev_index: absolute global index
	// (output) curr_index: absolute global index
	curr_index.clear();
	double M = 0;
	M = shrinkage / get_nr_variable();
	for (std::vector<int>::iterator it = prev_index.begin(); it != prev_index.end(); it++)
		if (w[*it] != 0 || fabs(loss_g[*it]) > 1 - M)
			curr_index.push_back(*it);
}

int l1r_fun::get_nr_variable(void)
{
	return prob->n;
}

double l1r_fun::smooth_line_search(double *w, double *smooth_step, double delta, double eta, std::vector<int> &index, double *fnew)
{
	double discard_threshold = 1e-4; //If the step size is too small then do not take this step
	double discard_threshold2 = 1e-6; //If the step size is too small then do not take this step
	int i;
	int l = prob->l;
	int inc = 1;
	double step_size = 1;
	int max_num_linesearch = 10;
	for (i=0;i<(int)index.size();i++)
		if (smooth_step[i] != 0)
		{
			double tmp = -w[index[i]] / smooth_step[i];
			if (tmp > 0)
				step_size = min(step_size, tmp);
		}
	if (step_size < discard_threshold)
	{
		info("INITIAL STEP SIZE TOO SMALL: %g\n",step_size);
		*fnew = current_f;
		return -1;
	}


	double reg_diff = 0;
	for (i=0;i<(int) index.size();i++)
		if (smooth_step[i] != 0)
			reg_diff += fabs(w[index[i]] + step_size * smooth_step[i]) - fabs(w[index[i]]);
	reg_diff /= step_size;

	Xv(smooth_step, z, &index[0], (int) index.size());

	int num_linesearch;
	delta *= eta;

	daxpy_(&l, &step_size, z, &inc, Xw, &inc);
	for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
	{
		if (step_size < discard_threshold2)
		{
			info("LINE SEARCH FAILED: Step size = %g\n",step_size);
			double factor = -step_size;
			daxpy_(&l, &factor, z, &inc, Xw, &inc);
			step_size = 0;
			*fnew = current_f;
			break;
		}
		double f_new = 0;
		for(i=0; i<l; i++)
			f_new += loss(i, Xw[i]);
		//If profiling gets sparsity, should consider tracking loss as a sum and only update individual losses
		mpi_allreduce(&f_new, 1, MPI_DOUBLE, MPI_SUM);
		f_new = f_new * C + reg + reg_diff * step_size;
		communication += 1 / global_n;
		if (f_new - current_f <= delta * step_size)
		{
			current_f = f_new;
			*fnew = f_new;
			reg += reg_diff * step_size;
			break;
		}
		else
		{
			step_size *= 0.5;
			double factor = -step_size;
			daxpy_(&l, &factor, z, &inc, Xw, &inc);
		}
	}
	if (num_linesearch == max_num_linesearch)
	{
		info("LINE SEARCH FAILED: Step size = %g\n",step_size);
		double factor = -step_size;
		daxpy_(&l, &factor, z, &inc, Xw, &inc);
		step_size = 0;
		*fnew = current_f;
	}

	return step_size;
}

double l1r_fun::armijo_line_search(double *step, double *w, double *loss_g, double *step_size, double eta, int *num_line_search_steps, double *delta_ret, int *index, int indexlength, int localstart, int locallength)
{
	int i;
	int inc = 1;
	int l = prob->l;
	int max_num_linesearch = 100;
	double localstepsize = (*step_size);

	double reg_diff = 0;

	int num_linesearch;
	double delta = 0;
	if (indexlength > 0)
	{
		for (i=0;i<locallength;i++)
			reg_diff += fabs(w[index[localstart + i]] + step[localstart + i]) - fabs(w[index[localstart + i]]);
		for (i=0;i<locallength;i++)
			delta += loss_g[index[localstart + i]] * step[localstart + i];
	}
	else
	{
		for (i=0;i<length;i++)
			reg_diff += fabs(w[start + i] + step[start + i]) - fabs(w[start + i]);
		for (i=0;i<length;i++)
			delta += loss_g[start + i] * step[start + i];
	}
	delta += reg_diff;
	mpi_allreduce(&delta, 1, MPI_DOUBLE, MPI_SUM);
	*delta_ret = delta;
	communication += 1 / global_n;
	delta *= eta;
	if (indexlength > 0)
		Xv(step, z, index, indexlength);
	else
		Xv(step, z);

	daxpy_(&l, &localstepsize, z, &inc, Xw, &inc);
	for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
	{
		double cond = 0;
		for(i=0; i<l; i++)
			cond += loss(i,Xw[i]);
		cond *= C;
		cond += reg_diff;
		//If profiling gets sparsity, should consider tracking loss as a sum and only update individual losses
		mpi_allreduce(&cond, 1, MPI_DOUBLE, MPI_SUM);
		communication += 1 / global_n;
		if (cond + reg - current_f <= delta * localstepsize)
		{
			mpi_allreduce(&reg_diff, 1, MPI_DOUBLE, MPI_SUM);
			communication += 1 / global_n;
			current_f = cond + reg;
			reg += reg_diff;
			break;
		}
		else
		{
			localstepsize *= 0.5;
			double factor = -localstepsize;
			daxpy_(&l, &factor, z, &inc, Xw, &inc);

			reg_diff = 0;
			if (indexlength > 0)
				for (i=0;i<locallength;i++)
					reg_diff += fabs(w[index[localstart + i]] + localstepsize * step[localstart + i]) - fabs(w[index[localstart + i]]);
			else
				for (i=0;i<length;i++)
					reg_diff += fabs(w[start + i] + localstepsize * step[start + i]) - fabs(w[start + i]);
		}
	}

	*num_line_search_steps = num_linesearch;
	*step_size = localstepsize;
	if (num_linesearch >= max_num_linesearch)
	{
		info("LINE SEARCH FAILED\n");
		*step_size = 0;
	}

	return current_f;
}

void l1r_fun::prox_grad(double *w, double *g, double *local_step, double *oldd, double alpha, int index_length)
{
	if (alpha <= 0)
	{
		int inc = 1;
		alpha = ddot_(&length, g, &inc, g, &inc);
		mpi_allreduce(&alpha, 1, MPI_DOUBLE, MPI_SUM);
		communication += 1 / global_n;
		alpha = sqrt(alpha);
	}
	if (index_length < 0)
		index_length = length;
	for (int i=0;i<index_length;i++)
	{
		double u = w[i] + oldd[i] - g[i] / alpha;
		local_step[i] = max(fabs(u) - 1.0 / alpha, 0.0);
		local_step[i] *= ((u > 0) - (u<0));
	}
}

l1r_lr_fun::l1r_lr_fun(const problem *prob, double C): l1r_fun(prob, C)
{
}

l1r_lr_fun::~l1r_lr_fun()
{
	delete[] z;
	delete[] Xw;
	delete[] D;
	delete[] recv_count;
	delete[] displace;
}

double l1r_lr_fun::loss(int i, double xw_i)
{
	double yXw = prob->y[i]*xw_i;
	if (yXw >= 0)
		return log(1 + exp(-yXw));
	else
		return (-yXw+log(1 + exp(yXw)));
}

void l1r_lr_fun::update_zD()
{
	for(int i=0;i<prob->l;i++)
	{
		z[i] = 1/(1 + exp(-prob->y[i]*Xw[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C*(z[i]-1)*prob->y[i];
	}
}


lasso::lasso(const problem *prob, double C): l1r_lr_fun(prob, C)
{
	for (int i=0;i<prob->l;i++)
		D[i] = 1.0;
}

lasso::~lasso()
{
}

double lasso::loss(int i, double xw_i)
{
	double tmp = (prob->y[i] - xw_i);
	return tmp * tmp / 2;
}

void lasso::update_zD()
{
	for(int i=0;i<prob->l;i++)
		z[i] = Xw[i] - prob->y[i];
}

l1r_l2_svc_fun::l1r_l2_svc_fun(const problem *prob, double C):
	l1r_lr_fun(prob, C)
{
}

l1r_l2_svc_fun::~l1r_l2_svc_fun()
{
}

double l1r_l2_svc_fun::loss(int i, double wx_i)
{
	double d = 1 - prob->y[i] * wx_i;
	if (d > 0)
		return C * d * d;
	else
		return 0;
}

void l1r_l2_svc_fun::update_zD()
{
	for(int i=0;i<prob->l;i++)
	{
		z[i] = Xw[i] * prob->y[i];
		if (z[i] < 1)
		{
			z[i] = 2 * C*prob->y[i]*(z[i]-1);
			D[i] = 2;
		}
		else
		{
			z[i] = 0;
			D[i] = 0;
		}
	}
}

grouplasso_mlr_fun::grouplasso_mlr_fun(const problem *prob, double C, int nr_class)
{
	this->C = C;
	this->nr_class = nr_class;
	this->prob = prob;
	int l=prob->l;
	Xw_length = l * nr_class;
	full_len = prob->n * nr_class;

	z = new double[Xw_length];
	Xw = new double[Xw_length];
	D = new double[Xw_length];
	this->C = C;

	int w_size = prob->n;
	int nr_node = mpi_get_size();
	int rank = mpi_get_rank();
	int shift = int(ceil(double(w_size) / double(nr_node)));
	recv_count = new int[nr_node];
	displace = new int[nr_node];
	localfeature_start = shift * rank;
	localfeatures = min(max(w_size - localfeature_start , 0), shift);
	if (localfeatures == 0)
	{
		localfeature_start = 0;
	}
	this->length = localfeatures * nr_class;
	this->start = localfeature_start * nr_class;
	shift *= nr_class;
	int counter = 0;
	for (int i=0;i<nr_node;i++)
	{
		recv_count[i] = shift;
		displace[i] = counter;
		counter += shift;
		if (counter >= full_len)
		{
			counter = 0;
			shift = 0;
		}
		else if (counter + shift > full_len)
			shift = full_len - counter;
	}
}

grouplasso_mlr_fun::~grouplasso_mlr_fun()
{
	delete[] z;
	delete[] Xw;
	delete[] D;
	delete[] recv_count;
	delete[] displace;
}

double grouplasso_mlr_fun::fun(double *w)
{
	int i,j;
	double f=0;
	int nnz = 0;
	int l=prob->l;
	int w_size = get_nr_variable();

	for (i=0;i<w_size;i+=nr_class)
	{
		int hit = 0;
		for (j=i;j<i+nr_class;j++)
			if (w[j] != 0)
			{
				hit = 1;
				break;
			}
		nnz += hit;
	}
	nnz *= nr_class;
	if (nnz == 0)
		memset(Xw, 0, sizeof(double) * l * nr_class);
	else if ((double)nnz / w_size < 0.5)
	{
		int *index = new int[nnz];
		double *subw = new double[nnz];
		int counter = 0;
		for (i=0;i<w_size;i+=nr_class)
		{
			int hit = 0;
			for (j=i;j<i+nr_class;j++)
				if (w[j] != 0)
				{
					hit = 1;
					break;
				}
			if (hit)
				for (j=i;j<i+nr_class;j++)
				{
					subw[counter] = w[j];
					index[counter++] = j;
				}
		}
		Xv(subw, Xw, index, nnz);
		delete[] index;
		delete[] subw;
	}
	else
		Xv(w, Xw);
	reg = regularizer(w + start, length);
	for(i=0;i<l;i++)
		f += loss(i, Xw + nr_class * i);
	double buffer[2] = {f, reg};
	mpi_allreduce(buffer, 2, MPI_DOUBLE, MPI_SUM);
	communication += 2 / global_n;

	reg = buffer[1];
	f = C * buffer[0] + reg;

	current_f = f;
	return(f);
}

double grouplasso_mlr_fun::f_update(double *w, double *step, double Q, double eta, int *index, int index_length, int localstart, int locallength)
{
	if (Q > 0)
		return current_f;
	int *localindex = NULL;
	double *localstep;
	double *localw;
	int i,j;
	int inc = 1;
	int l = prob->l;
	double one = 1.0;
	int nnz = 0;
	int len = index_length;
	int w_size = get_nr_variable();
	if (len == 0)
		len = w_size;


	double reg_diff = 0;

	if (index != NULL) // If a subset is selected
	{
		localindex = index + localstart; // Assume that localindex is already grouped according to features, so the length is a multiple of nr_class
		localstep = step + localstart;
		for (i=0;i<locallength;i+=nr_class)
		{
			double oldreg = 0;
			double newreg = 0;
			for (int j=i;j<i+nr_class;j++)
			{
				oldreg += w[localindex[j]] * w[localindex[j]];
				double tmp = w[localindex[j]] + localstep[j];
				newreg += tmp * tmp;
			}
			reg_diff += sqrt(newreg) - sqrt(oldreg);
		}
	}
	else
	{
		localw = w + start;
		localstep = step + start;
		for (i=0;i<length;i+=nr_class)
		{
			double oldreg = 0;
			double newreg = 0;
			for (int j=i;j<i+nr_class;j++)
			{
				oldreg += localw[j] * localw[j];
				double tmp = localw[j] + localstep[j];
				newreg += tmp * tmp;
			}
			reg_diff += sqrt(newreg) - sqrt(oldreg);
		}
	}
	for (i=0;i<len;i+=nr_class)
	{
		int hit = 0;
		for (j=i;j<i+nr_class;j++)
			if (step[j] != 0)
			{
				hit = 1;
				break;
			}
		nnz += hit;
	}
	nnz *= nr_class;
	if (nnz == 0)
		memset(z, 0, sizeof(double) * l * nr_class);
	else if ((double)nnz / len < 0.5)
	{
		int *subindex = new int[nnz];
		double *substep = new double[nnz];
		int counter = 0;
		if (index_length == 0)
			for (i=0;i<len;i+=nr_class)
			{
				int hit = 0;
				for (j=i;j<i+nr_class;j++)
					if (step[j] != 0)
					{
						hit = 1;
						break;
					}
				if (hit)
					for (j=i;j<i+nr_class;j++)
					{
						substep[counter] = step[j];
						subindex[counter++] = j;
					}
			}
		else
			for (i=0;i<index_length;i+=nr_class)
			{
				int hit = 0;
				for (j=i;j<i+nr_class;j++)
					if (step[j] != 0)
					{
						hit = 1;
						break;
					}
				if (hit)
					for (j=i;j<i+nr_class;j++)
					{
						substep[counter] = step[j];
						subindex[counter++] = index[j];
					}
			}
		Xv(substep, z, subindex, nnz);
		delete[] subindex;
		delete[] substep;
	}
	else
		Xv(step, z, index, index_length);

	daxpy_(&Xw_length, &one, z, &inc, Xw, &inc);
	double f_new = 0;
	for(i=0; i<l; i++)
		f_new += loss(i, Xw + nr_class * i);
	f_new *= C;
	f_new += reg_diff;
	//If profiling gets sparsity, should consider and only updating individual losses
	//This should be the case when data has sparsity and the update is extremely sparse
	mpi_allreduce(&f_new, 1, MPI_DOUBLE, MPI_SUM);
	communication += 1 / global_n;
	f_new += reg;
	if (f_new - current_f <= eta * Q)
	{
		mpi_allreduce(&reg_diff, 1, MPI_DOUBLE, MPI_SUM);
		communication += 1 / global_n;
		current_f = f_new;
		reg += reg_diff;
	}
	else
	{
		double factor = -1;
		daxpy_(&Xw_length, &factor, z, &inc, Xw, &inc);
	}

	return current_f;
}

void grouplasso_mlr_fun::loss_grad(double *w, double *g, const std::vector<int> &fullindex)
{
	int i;
	int index_length = (int)fullindex.size();
	int w_size=get_nr_variable();
	if (fullindex[0] != -1)
		w_size = index_length;
	update_zD();

	memset(g, 0, sizeof(double) * w_size);
	XTv(z, g, fullindex);
	
	mpi_allreduce(g, w_size, MPI_DOUBLE, MPI_SUM);
	if (fullindex[0] == -1)
		communication += 1.0;
	else
	{
		for (i = index_length- 1; i >= 0; i--)
		{
			if (fullindex[i] == i)
				break;
			g[fullindex[i]] = g[i];
			g[i] = 0;
		}
		communication += (double) index_length / global_n;
	}
}

void grouplasso_mlr_fun::Hv(double *s, double *Hs, const std::vector<int> &fullindex, double *w)
{
	int inc = 1;
	int i,j;
	int w_size=(int)fullindex.size();
	double tmp;

	int localstart, locallength;
	get_local_index_start_and_length(w_size, &localstart, &locallength);
	//The part of the Hessian from the regularizer

	Xv(s, z, &fullindex[0], w_size);

	for(i=0;i<Xw_length;i+=nr_class)
	{
		tmp = 0;
		for (j=i;j<i + nr_class;j++)
			tmp += z[j] * D[j];
		for (j=i;j<i + nr_class;j++)
			z[j] = C * D[j] * (z[j] - tmp);
	}
	XTv(z, Hs, fullindex);

	for (i=0;i<locallength;i+=nr_class)
	{
		double *localw = w + fullindex[localstart + i];
		double *localHs = Hs + localstart + i;
		double *locals = s + localstart + i;
		double norm = dnrm2_(&nr_class, localw, &inc);
		double prod = ddot_(&nr_class, locals, &inc, localw, &inc);
		double scale = -prod / (norm * norm * norm);
		daxpy_(&nr_class, &scale, localw, &inc, localHs, &inc);
		scale = 1.0 / norm;
		daxpy_(&nr_class, &scale, locals, &inc, localHs, &inc);
	}
	mpi_allreduce(Hs, w_size, MPI_DOUBLE, MPI_SUM);
	communication += w_size / global_n;
}

double grouplasso_mlr_fun::vHv(double *s, const std::vector<int> &index)
{
	int i,j;
	int inc = 1;

	if (index[0] == -1)
		Xv(s, z);
	else
		Xv(s, z, index.data(), (int) index.size());
	double alpha = 0;
	for(i=0;i<Xw_length;i+=nr_class)
	{
		double tmp = 0;
		for (j=i;j<i + nr_class;j++)
		{
			alpha += z[j] * D[j] * z[j];
			tmp += z[j] * D[j];
		}
		alpha -= tmp * tmp;
	}

	double buffer[2];
	double norm2;
	if (index[0] == -1)
		norm2 = ddot_(&length, s + start, &inc, s + start, &inc);
	else
	{
		int index_size = (int) index.size();
		int nr_node = mpi_get_size();
		int rank = mpi_get_rank();
		int shift = int(ceil(double(index_size) / double(nr_node)));
		int indexstart = shift * rank;
		int indexlength = min(max(index_size - indexstart, 0), shift);
		if (indexlength == 0)
			indexstart = 0;
		norm2 = ddot_(&indexlength, s + indexstart, &inc, s + indexstart, &inc);
	}
	buffer[0] = alpha;
	buffer[1] = norm2;
	mpi_allreduce(buffer, 2, MPI_DOUBLE, MPI_SUM);
	norm2 = buffer[1];
	alpha = buffer[0] * C / norm2;
	communication += 2 / global_n;
	return alpha;
}

void grouplasso_mlr_fun::get_diag_preconditioner(double *M, const std::vector<int> &fullindex, double *w)
{
	int inc = 1;
	int i,j;
	int w_size=(int)fullindex.size();
	feature_node **x = prob->x;
	for (i=0; i<w_size; i++)
		M[i] = 0.0;

	int localstart, locallength;
	get_local_index_start_and_length(w_size, &localstart, &locallength);
	for (i=0;i<locallength;i+=nr_class)
	{
		double *localw = w + fullindex[localstart + i];
		double *localM = M + localstart + i;
		double norm = dnrm2_(&nr_class, localw, &inc);
		for (j=0;j<nr_class;j++)
			localM[j] = (1.0 - localw[j] * localw[j] / (norm * norm)) / norm;
	}

	for (i=0; i<w_size; i+=nr_class)
	{
		feature_node *s;
		s=x[fullindex[i] / nr_class];
		while (s->index!=-1)
		{
			double tmp = s->value*s->value;
			for (j=0;j<nr_class;j++)
			{
				double p = D[(s->index - 1) * nr_class + j];
				M[i + j] += tmp*C*(p * (1 - p));
			}
			s++;
		}
	}
	mpi_allreduce(M, w_size, MPI_DOUBLE, MPI_SUM);
	communication += w_size / global_n;
}

void grouplasso_mlr_fun::full_grad(double *w, double *g, std::vector<int> &index, double *full_g, std::vector<int> &nonzero_set)
{
	int inc = 1;
	for (int i=0, full_i=0;i<(int) index.size();i+=nr_class)
	{
		double norm = dnrm2_(&nr_class, w + index[i], &inc);
		if (norm > 0)
			for (int j=index[i];j<index[i] + nr_class;j++)
			{
				nonzero_set.push_back(j);
				full_g[full_i++] = g[j] + w[j] / norm;
			}
	}
}

void grouplasso_mlr_fun::setselection(double *w, double *loss_g, std::vector<int> &prev_index, std::vector<int> &curr_index, double shrinkage)
{
	// (input)  prev_index: absolute global index
	// (output) curr_index: absolute global index
	int inc = 1;
	curr_index.clear();
	double M = 0;
	M = shrinkage / double(get_nr_variable() / nr_class);
	for (int i=0;i<(int)prev_index.size();i+=nr_class)
	{
		double norm = dnrm2_(&nr_class, w + prev_index[i], &inc);
		double gnorm = dnrm2_(&nr_class, loss_g + prev_index[i], &inc);
		if (norm > 0 || gnorm > 1 - M)
			for (int j=prev_index[i];j<prev_index[i] + nr_class;j++)
				curr_index.push_back(j);
	}
}

int grouplasso_mlr_fun::get_nr_variable()
{
	return full_len;
}

double grouplasso_mlr_fun::armijo_line_search(double *step, double *w, double *loss_g, double *step_size, double eta, int *num_line_search_steps, double *delta_ret, int *index, int indexlength, int localstart, int locallength)
{
	int i;
	int inc = 1;
	int l = prob->l;
	int max_num_linesearch = 100;
	double localstepsize = (*step_size);

	int *localindex = NULL;
	double *localstep;
	double *localw = NULL;

	int num_linesearch;
	double delta = 0;

	double reg_diff = 0;

	if (indexlength > 0)
	{
		localindex = index + localstart;
		localstep = step + localstart;
		for (i=0;i<locallength;i+=nr_class)
		{
			double oldreg = 0;
			double newreg = 0;
			for (int j=i;j<i+nr_class;j++)
			{
				oldreg += w[localindex[j]] * w[localindex[j]];
				double tmp = w[localindex[j]] + localstep[j];
				newreg += tmp * tmp;
				delta += loss_g[localindex[j]] * localstep[j];
			}
			reg_diff += sqrt(newreg) - sqrt(oldreg);
		}
	}
	else
	{
		localw = w + start;
		localstep = step + start;
		double *localg = loss_g + start;
		for (i=0;i<length;i+=nr_class)
		{
			double oldreg = 0;
			double newreg = 0;
			for (int j=i;j<i+nr_class;j++)
			{
				oldreg += localw[j] * localw[j];
				double tmp = localw[j] + localstep[j];
				newreg += tmp * tmp;
				delta += localg[j] * localstep[j];
			}
			reg_diff += sqrt(newreg) - sqrt(oldreg);
		}
	}

	delta += reg_diff;
	mpi_allreduce(&delta, 1, MPI_DOUBLE, MPI_SUM);
	*delta_ret = delta;
	communication += 1 / global_n;
	delta *= eta;
	if (indexlength > 0)
		Xv(step, z, index, indexlength);
	else
		Xv(step, z);

	daxpy_(&Xw_length, &localstepsize, z, &inc, Xw, &inc);
	for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
	{
		double cond = 0;
		for(i=0; i<l; i++)
			cond += loss(i, Xw + nr_class * i);;
		cond *= C;
		cond += reg_diff;
		//If profiling gets sparsity, should consider tracking loss as a sum and only update individual losses
		mpi_allreduce(&cond, 1, MPI_DOUBLE, MPI_SUM);
		communication += 1 / global_n;
		if (cond + reg - current_f <= delta * localstepsize)
		{
			mpi_allreduce(&reg_diff, 1, MPI_DOUBLE, MPI_SUM);
			communication += 1 / global_n;
			current_f = cond + reg;
			reg += reg_diff;
			break;
		}
		else
		{
			localstepsize *= 0.5;
			double factor = -localstepsize;
			daxpy_(&Xw_length, &factor, z, &inc, Xw, &inc);

			reg_diff = 0;
			if (indexlength > 0)
			{
				for (i=0;i<locallength;i+=nr_class)
				{
					double oldreg = 0;
					double newreg = 0;
					for (int j=i;j<i+nr_class;j++)
					{
						oldreg += w[localindex[j]] * w[localindex[j]];
						double tmp = w[localindex[j]] + localstepsize * localstep[j];
						newreg += tmp * tmp;
					}
					reg_diff += sqrt(newreg) - sqrt(oldreg);
				}
			}
			else
			{
				for (i=0;i<length;i+=nr_class)
				{
					double oldreg = 0;
					double newreg = 0;
					for (int j=i;j<i+nr_class;j++)
					{
						oldreg += localw[j] * localw[j];
						double tmp = localw[j] + localstepsize * localstep[j];
						newreg += tmp * tmp;
					}
					reg_diff += sqrt(newreg) - sqrt(oldreg);
				}
			}
		}
	}

	*num_line_search_steps = num_linesearch;
	*step_size = localstepsize;
	if (num_linesearch >= max_num_linesearch)
	{
		info("LINE SEARCH FAILED\n");
		*step_size = 0;
	}

	return current_f;
}

double grouplasso_mlr_fun::smooth_line_search(double *w, double *smooth_step, double delta, double eta, std::vector<int> &index, double *fnew)
{
	double discard_threshold = 1e-8; //If the step size is too small then do not take this step
	double discard_threshold2 = 1e-9; //If the step size is too small then do not take this step
	int i,j;
	int l = prob->l;
	int inc = 1;
	double step_size = 1;
	int max_num_linesearch = 30;
	for (i=0;i<(int)index.size();i+=nr_class)
	{
		double proportion;
		if (smooth_step[i] == 0)
			continue;
		else
			proportion = -w[index[i]] / smooth_step[i];
		if (proportion > 0)
			for (j=i+1;j<i+nr_class;j++)
			{
				if (smooth_step[j] == 0)
					continue;
				double current_proportion = -w[index[j]] / smooth_step[j];
				if (current_proportion < 0 || fabs(current_proportion - proportion) > 1e-20)
				{
					proportion = -1;
					break;
				}
			}
		if (proportion > 0)
			step_size = min(step_size, proportion);
	}
	if (step_size < discard_threshold)
	{
		info("INITIAL STEP SIZE TOO SMALL: %g\n",step_size);
		*fnew = current_f;
		return -1;
	}

	Xv(smooth_step, z, &index[0], (int) index.size());

	int num_linesearch;
	delta *= eta;

	daxpy_(&Xw_length, &step_size, z, &inc, Xw, &inc);
	for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
	{
		if (step_size < discard_threshold2)
		{
			info("LINE SEARCH FAILED: Step size = %g\n",step_size);
			double factor = -step_size;
			daxpy_(&Xw_length, &factor, z, &inc, Xw, &inc);
			step_size = 0;
			*fnew = current_f;
			break;
		}
		double f_new = 0;
		for(i=0; i<l; i++)
			f_new += loss(i, Xw + nr_class * i);
		mpi_allreduce(&f_new, 1, MPI_DOUBLE, MPI_SUM);
		double reg_diff = 0;
		for (i=0;i<(int) index.size();i+=nr_class)
		{
				double oldreg = 0;
				double newreg = 0;
				for (int j=i;j<i+nr_class;j++)
				{
					oldreg += w[index[j]] * w[index[j]];
					double tmp = w[index[j]] + step_size * smooth_step[j];
					newreg += tmp * tmp;
				}
				reg_diff += sqrt(newreg) - sqrt(oldreg);
		}
		
		f_new = f_new * C + reg + reg_diff;
		communication += 1 / global_n;
		if (f_new - current_f <= delta * step_size)
		{
			current_f = f_new;
			*fnew = f_new;
			reg += reg_diff;
			break;
		}
		else
		{
			step_size *= 0.5;
			double factor = -step_size;
			daxpy_(&Xw_length, &factor, z, &inc, Xw, &inc);
		}
	}
	if (num_linesearch == max_num_linesearch)
	{
		info("LINE SEARCH FAILED: Step size = %g\n",step_size);
		double factor = -step_size;
		daxpy_(&l, &factor, z, &inc, Xw, &inc);
		step_size = 0;
		*fnew = current_f;
	}

	return step_size;
}

void grouplasso_mlr_fun::prox_grad(double *w, double *g, double *local_step, double *oldd, double alpha, int index_length)
{
	if (alpha <= 0)
	{
		int inc = 1;
		alpha = ddot_(&length, g, &inc, g, &inc);
		mpi_allreduce(&alpha, 1, MPI_DOUBLE, MPI_SUM);
		communication += 1 / global_n;
		alpha = sqrt(alpha);
	}
	if (index_length < 0)
		index_length = length;

	double coeff = -1.0 / alpha;
	double one = 1.0;
	int inc = 1;
	memcpy(local_step, oldd, sizeof(double) * index_length);
	daxpy_(&index_length, &one, w, &inc, local_step, &inc);
	daxpy_(&index_length, &coeff, g, &inc, local_step, &inc);
	for (int i=0;i<index_length;i+=nr_class)
	{
		double norm = dnrm2_(&nr_class, local_step + i, &inc);
		double factor = norm + coeff;
		if (factor <= 0)
			memset(local_step + i, 0, sizeof(double) * nr_class);
		else
		{
			factor = factor / (factor - coeff);
			dscal_(&nr_class, &factor, local_step + i, &inc);
		}
	}
}

double grouplasso_mlr_fun::regularizer(double *w, int w_length)
{
	int inc = 1;
	double ret = 0;
	for (int i=0;i<w_length;i+=nr_class)
		ret += dnrm2_(&nr_class, w + i, &inc);
	return ret;
}

void grouplasso_mlr_fun::get_local_index_start_and_length(int global_length, int *local_start, int *local_length)
{
	int nr_node = mpi_get_size();
	int rank = mpi_get_rank();
	int shift = int(ceil(double(global_length / nr_class) / double(nr_node))) * nr_class;
	int idx_start = shift * rank;
	int idx_length = min(max(global_length - idx_start, 0), shift);
	if (idx_length == 0)
		idx_start = 0;
	*local_start = idx_start;
	*local_length = idx_length;
}

void grouplasso_mlr_fun::Xv(double *v, double *Xv, const int *index, int index_length)
{
	int i,j;
	int w_size=get_nr_variable();
	if (index_length > 0)
		w_size = index_length;
	feature_node **x=prob->x;
	memset(Xv, 0, sizeof(double) * Xw_length);

	for (i=0;i<w_size;i+=nr_class)
	{
		feature_node *s;
		int idx1;
		if (index_length > 0)
			idx1 = index[i];
		else
			idx1 = i;
		s = x[idx1 / nr_class];
		while(s->index != -1)
		{
			int idx2 = (s->index - 1) * nr_class;
			for (j=0;j<nr_class;j++)
				Xv[idx2 + j] += v[i + j]*s->value;
			s++;
		}
	}
}

void grouplasso_mlr_fun::XTv(double *v, double *XTv, const std::vector<int> &fullindex)
{
	int i,j;
	int w_size=(int)fullindex.size();
	if (fullindex[0] == -1)
		w_size = get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i+=nr_class)
	{
		feature_node *s;
		int idx1;
		if (fullindex[0] != -1)
			idx1 = fullindex[i];
		else
			idx1 = i;
		s = x[idx1 / nr_class];
		while(s->index != -1)
		{
			int idx2 = (s->index - 1) * nr_class;
			for (j=0;j<nr_class;j++)
				XTv[i + j] += v[idx2 + j]*s->value;
			s++;
		}
	}
}

double grouplasso_mlr_fun::loss(int i, double *wx_i)
{
	//The loss is log[ (sum_j exp^{wx_i[j]}) / exp^{wx_i[y_i]} ]
	double ret = 0;
	double max_val = wx_i[0];
	for (int j=1;j<nr_class;j++)
		max_val = max(max_val, wx_i[j]);
	for (int j=0;j<nr_class;j++)
		ret += exp(wx_i[j] - max_val);
	return (log(ret) + max_val - wx_i[(int)prob->y[i]]);
}

void grouplasso_mlr_fun::update_zD()
{
	int inc = 1;
	for(int i=0;i<Xw_length;i+=nr_class)
	{
		double max_val = Xw[i];
		double normalizer = 0;
		for (int j=i+1;j<i+nr_class;j++)
			max_val = max(max_val, Xw[j]);
		for (int j=i;j<i+nr_class;j++)
		{
			z[j] = exp(Xw[j] - max_val);
			normalizer += z[j];
		}
		normalizer = 1.0 / normalizer;
		dscal_(&nr_class, &normalizer, z + i, &inc);
	}
	memcpy(D, z, sizeof(double)* Xw_length);
	for(int i=0;i<prob->l;i++)
		z[i * nr_class + (int) prob->y[i]] -= 1;
}

// begin of solvers

class SOLVER
{
public:
	SOLVER(const regularized_fun *fun_obj, double eps = 0.1, double eta = 1e-4, int max_iter = 10000);
	~SOLVER();
	void set_print_string(void (*i_print) (const char *buf));

protected:
	double eps;
	double eta;
	int max_iter;
	int start;
	int length;
	int *recv_count;
	int *displace;
	regularized_fun *fun_obj;
	void info(const char *fmt,...);
	void (*solver_print_string)(const char *buf);
};

class MADPQN: public SOLVER
{
public:
	MADPQN(const regularized_fun *fun_obj, double eps = 0.0001, int m=10, double inner_eps = 0.01, int max_inner = 100, double eta = 1e-4, int max_iter = 1000000);
	~MADPQN();

	void madpqn(double *w, bool disable_smooth);
protected:
	int M;
	double inner_eps;
	int max_inner;
	void update_inner_products(double **inner_product_matrix, int k, int DynamicM, double *s, double *y, const std::vector<int> &index = std::vector<int>{-1});
	void compute_R(double *R, int DynamicM, double **inner_product_matrix, int k, double gamma);
private:
	double SpaRSA(double *w, double *loss_g, double *R, double *s, double *y, double gamma, double *local_step, double scaling, int DynamicM, std::vector<int> &index);
	double newton(double *g, double *step, const std::vector<int> &global_nonzero_set, int max_cg_iter, double *w = NULL);
	double Q_prox_grad(double *step, double *w, double *g, int global_length, int local_length, int local_start, int *index, double alpha, double *tmps, double *f, int *counter_ret);
};

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

void SOLVER::info(const char *fmt,...)
{
	if(mpi_get_rank()!=0)
		return;
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*solver_print_string)(buf);
}

void SOLVER::set_print_string(void (*print_string) (const char *buf))
{
	solver_print_string = print_string;
}

SOLVER::SOLVER(const regularized_fun *fun_obj, double eps, double eta, int max_iter)
{
	this->fun_obj=const_cast<regularized_fun *>(fun_obj);
	this->eps=eps;
	this->max_iter=max_iter;
	this->eta = eta;
	solver_print_string = default_print;
	this->recv_count = fun_obj->recv_count;
	this->displace = fun_obj->displace;
	this->start = fun_obj->start;
	this->length = fun_obj->length;
}
SOLVER::~SOLVER()
{
}


enum Stage {initial, alternating, smooth};

MADPQN::MADPQN(const regularized_fun *fun_obj, double eps, int m, double inner_eps, int max_inner, double eta, int max_iter):
	SOLVER(fun_obj, eps, eta, max_iter)
{
	this->M = m;
	this->inner_eps = inner_eps;
	this->max_inner = max_inner;
}

MADPQN::~MADPQN()
{
}

void fill_range(std::vector<int> &v, int n)
{
	v.clear();
	for (int i = 0; i < n; i++)
		v.push_back(i);
}

void MADPQN::madpqn(double *w, bool disable_smooth)
{
	const double update_eps = 1e-10;//Ensure PD of the LBFGS matrix
	const double sparse_factor = 0.4;//The sparsity threshold for deciding when we should switch to sparse communication
	const int max_modify_Q = 20;
	const int smooth_trigger = 10;
	const double ALPHA_MAX = 1e10;
	const double ALPHA_MIN = 1e-4;
	const int min_inner = 2;
	int inc = 1;
	double one = 1.0;
	double minus_one = -1.0;

	int n = fun_obj->get_nr_variable();

	int i, k = 0;
	int inner_iter = 0, total_iter = 0;
	int skip = 0;
	int DynamicM = 0;
	double s0y0 = 0;
	double s0s0 = 0;
	double y0y0 = 0;
	double f;
	double Q0;
	double Q = 0;
	double eps1 = max(eps,0.0001);
	double min_eps1 = min(eps, eps1);
	double gamma = ALPHA_MAX;
	int skip_flag;
	int64_t timer_st, timer_ed;
	double accumulated_time = 0;
	double all_reduce_buffer[3];
	double alpha;
	int nr_node = mpi_get_size();
	int idx_start = start, idx_length = length;

	int *tmprecvcount = new int[nr_node];
	int *tmpdisplace = new int[nr_node];
	double *s = new double[2*M*length];
	double *y = s + M*length;
	double *tmpy = new double[length];
	double *tmps = new double[length];
	double *loss_g = new double[n];
	double *local_step = new double[length];
	double *step = new double[n];
	double *full_g = new double[n];
	double *R = new double[4 * M * M];
	double **inner_product_matrix = new double*[M];
	double fnew;
	int counter = 0;
	int ran_smooth_flag = 0; //flag on if last step was a smooth optimization step
	Stage stage = initial;
	int latest_index_size = 0;
	int sparse_communication_flag = 0;
	int init_max_cg_iter = 5;
	double cg_increment = 2;
	int current_max_cg_iter = init_max_cg_iter;
	int search = 1;

	for (i=0; i < M; i++)
		inner_product_matrix[i] = new double[2*M];

	// for alternating between previous and current indices
	int prev_idx, curr_idx;
	std::vector<int> index[2], local_index;
	int global_index_length = n, index_length = length;
	int unchanged_counter = 0;

	global_n = (double)n;

	// calculate the Q quantity used in update acceptance with the step being a
	// proximal gradient one, at w=0 for stopping condition.
	double *w0 = new double[n];
	for (i=0; i<n; i++)
		w0[i] = 0;
	f = fun_obj->fun(w0);
	fun_obj->loss_grad(w0, loss_g);
	alpha = min(max(fabs(fun_obj->vHv(loss_g)), ALPHA_MIN), ALPHA_MAX);

	// initialize with absolute global index
	fill_range(index[0], n);
	fill_range(index[1], n);
	curr_idx = inner_iter % 2;
	prev_idx = (inner_iter + 1) % 2;
	Q = INF;

	Q0 = Q_prox_grad(step, w0, loss_g, n, length, start, &index[0][0], alpha, tmps, &f, &counter);
	communication = 0;
	if (Q0 == 0)
	{
		info("WARNING: Q0=0\n");
		search = 0;
	}
	delete [] w0;

	f = fun_obj->fun(w);
	fnew = f;
	timer_st = wall_clock_ns();

	while (total_iter < max_iter && search)
	{
		skip_flag = 0;
		timer_ed = wall_clock_ns();
		accumulated_time += wall_time_diff(timer_ed, timer_st);
		int nnz = 0;
		for (i = 0; i < length; i++)
			if (w[start + i] != 0)
				nnz++;
		mpi_allreduce(&nnz, 1, MPI_INT, MPI_SUM);
		if (Q == 0)
		{
			info("WARNING: Update Failed\n");
			if (total_iter <= 1)
				break;
		}
		else
			info("iter=%d m=%d f=%15.20e Q=%g subprobs=%d w_nnz=%d active_set=%d elapsed_time=%g communication=%g\n", total_iter, DynamicM,  f,  Q, counter, nnz, (int)index[curr_idx].size(), accumulated_time,communication);
		timer_st = wall_clock_ns();
		
		double stopping = Q / Q0;
		if (total_iter > 0 && stopping < eps1)
		{
			if ((int) index[curr_idx].size() == n)
			{
				if (stopping <= eps)
					break;
				else
					eps1 = max(0.001 * eps1, min_eps1);
			}
			if (inner_iter >= min_inner || Q == 0)
			{
				DynamicM = 0;
				inner_iter = 0;
				skip = 0;
				k = 0;
				fill_range(index[0], n);
				fill_range(index[1], n);
				local_index.clear();
				eps1 = max(0.001 * eps1, min_eps1);
				index_length = 0;
				global_index_length = n;
				idx_start = start;
				idx_length = length;
			}
			fun_obj->loss_grad(w, loss_g);
		}
		else
		{
			fun_obj->loss_grad(w, loss_g, index[curr_idx]);
		}
		
		// Decide if we want to add the new pair of s,y, then update the inner products if added

		if (inner_iter != 0)
		{
			if ((int)index[curr_idx].size() == n)
			{
				daxpy_(&length, &one, loss_g + start, &inc, tmpy, &inc);
				all_reduce_buffer[0] = ddot_(&length, tmpy, &inc, tmps, &inc);
				all_reduce_buffer[1] = ddot_(&length, tmps, &inc, tmps, &inc);
				all_reduce_buffer[2] = ddot_(&length, tmpy, &inc, tmpy, &inc);
			}
			else
			{
				memset(all_reduce_buffer, 0, sizeof(double) * 3);
				for (std::vector<int>::iterator it = local_index.begin(); it != local_index.end(); it++)
				{
					i = *it;
					tmpy[i] += loss_g[start + i];
					all_reduce_buffer[0] += tmpy[i] * tmps[i];
					all_reduce_buffer[1] += tmps[i] * tmps[i];
					all_reduce_buffer[2] += tmpy[i] * tmpy[i];
				}
			}
			
			mpi_allreduce(all_reduce_buffer, 3, MPI_DOUBLE, MPI_SUM);
			communication += 3 / global_n;
			s0y0 = all_reduce_buffer[0];
			s0s0 = all_reduce_buffer[1];
			y0y0 = all_reduce_buffer[2];
			if (s0y0 >= update_eps * s0s0)
			{
				memcpy(y + (k*length), tmpy, length * sizeof(double));
				memcpy(s + (k*length), tmps, length * sizeof(double));
				gamma = y0y0 / s0y0;
			}
			else
			{
				info("Skipped updating s and y\n");
				skip_flag = 1;
				skip++;
			}
			DynamicM = min(inner_iter - skip, M);
		}

		if (!ran_smooth_flag)
		{
			int	tmp_idx = curr_idx;
			curr_idx = prev_idx;
			prev_idx = tmp_idx;
		}

		memset(local_step, 0, sizeof(double) * length);
		memset(tmps, 0, sizeof(double) * length);
		if (DynamicM > 0)
		{
			if (!skip_flag)
			{
				if (index_length < length * sparse_factor && index_length != 0)
					update_inner_products(inner_product_matrix, k, DynamicM, s, y, local_index);
				else
					update_inner_products(inner_product_matrix, k, DynamicM, s, y);

				compute_R(R, DynamicM, inner_product_matrix, k, gamma);
				k = (k+1)%M;
			}
			alpha = 1;
		}
		else
			alpha = min(max(min(fabs(fun_obj->vHv(loss_g, index[curr_idx])), gamma), ALPHA_MIN), ALPHA_MAX);
		if (total_iter > 0)
		{
			if (unchanged_counter >= smooth_trigger)
			{
				if (stage == initial)
					stage = alternating;
			}
			else
				stage = initial;
		}
		if (inner_iter > 0 && !ran_smooth_flag)
		{
			fun_obj->setselection(w, loss_g, index[prev_idx], index[curr_idx], min(Q / Q0,1.0));
			global_index_length = (int)index[curr_idx].size();
			if (global_index_length == latest_index_size) // Note here we only checked the size of the index set remained unchanged, but didn't really check whether the elements changed or not
				unchanged_counter++;
			else
				unchanged_counter = 0;
			latest_index_size = global_index_length;

			local_index.clear();
			for (std::vector<int>::iterator it = index[curr_idx].begin(); it != index[curr_idx].end(); it++)
			{
				if (*it >= start + length)
					break;
				if (*it >= start)
					local_index.push_back(*it - start);
			}
			index_length = (int) local_index.size();
			fun_obj->get_local_index_start_and_length(global_index_length, &idx_start, &idx_length);
		}

		sparse_communication_flag = (global_index_length < n * sparse_factor);
		memcpy(tmpy, loss_g + start, sizeof(double) * length);
		dscal_(&length, &minus_one, tmpy, &inc);
		if (!disable_smooth && (stage == smooth && ran_smooth_flag))
		{
			ran_smooth_flag = 0;
			alpha = min(max(min(fabs(fun_obj->vHv(loss_g, index[curr_idx])), gamma), ALPHA_MIN), ALPHA_MAX);
			Q = Q_prox_grad(step, w, loss_g, global_index_length, idx_length, idx_start, &index[curr_idx][0], alpha, tmps, &f, &counter);
			if (Q < 0)
			{
				inner_iter++;
				total_iter++;
			}
			
			continue;
		}
		else if (!disable_smooth && (inner_iter > 0 && stage != initial && !ran_smooth_flag))
		{
			// all machines conduct the same CG procedure
			std::vector<int> global_nonzero_set;
			global_nonzero_set.clear();
			fun_obj->full_grad(w, loss_g, index[curr_idx], full_g, global_nonzero_set);
			int global_nnz_size = (int)global_nonzero_set.size();
			int max_cg_iter = min(global_nnz_size, current_max_cg_iter);
			if (global_nnz_size < global_index_length || stage == alternating)
			{
				max_cg_iter = init_max_cg_iter;
				current_max_cg_iter = init_max_cg_iter;
				stage = alternating;
			}
			double delta = newton(full_g, step, global_nonzero_set, max_cg_iter, w);
			double step_size;
			if (delta >= 0)
				step_size = 0;
			else
				step_size = fun_obj->smooth_line_search(w, step, delta, eta, global_nonzero_set, &fnew);

			if (step_size == 1)
			{
				stage = smooth;
				current_max_cg_iter = int(cg_increment * current_max_cg_iter);
			}
			else
			{
				stage = alternating;
				current_max_cg_iter = init_max_cg_iter;
			}
			if (step_size <= 0)
			{
				unchanged_counter = 0;
				ran_smooth_flag = 0;
			}
			else
			{
				info("step_size = %g\n",step_size);
				f = fnew;
				for (i=0;i<global_nnz_size;i++)
					w[global_nonzero_set[i]] += step_size * step[i];
				for (i=0;i<global_nnz_size;i++)
				{
					int idx = global_nonzero_set[i];
					if (idx >= start + length)
						break;
					if (idx >= start)
						tmps[idx - start] = step_size * step[i];
				}

				ran_smooth_flag = 1;
				info("**Smooth Step\n");
				Q = delta;
				counter = 0;
				inner_iter++;
				total_iter++;
				continue;
			}
		}
		//First decide the size to communicate from each machine
		//Then communication to gather the sparse update
		//cost: 1 rounds of sparse communication as we have the full index on all machines
		MPI_Allgather(&index_length, 1, MPI_INT, tmprecvcount, 1, MPI_INT, MPI_COMM_WORLD);
		communication += nr_node / (double) n;

		if (sparse_communication_flag) // conduct sparse communication
		{
			tmpdisplace[0] = 0;
			for (i=1;i<nr_node;i++)
				tmpdisplace[i] = tmpdisplace[i-1] + tmprecvcount[i-1];
		}
		else
			memset(step, 0, sizeof(double) * n);


		ran_smooth_flag = 0;
		counter = 0;

		if (DynamicM == 0)
		{
			Q = Q_prox_grad(step, w, loss_g, global_index_length, idx_length, idx_start, &index[curr_idx][0], alpha, tmps, &f, &counter);
			if (Q < 0)
			{
				inner_iter++;
				total_iter++;
			}
		}
		else
		{
			do
			{
				Q = SpaRSA(w + start, loss_g + start, R, s, y, gamma, local_step, alpha, DynamicM, local_index);
				if (sparse_communication_flag)
				{
					//First decide the size to communicate from each machine
					//Then communication to gather the sparse update
					//cost: 1 rounds of sparse communication as we have the full index on all machines
					int rank = mpi_get_rank();
					MPI_Allgatherv(local_step, (int) index_length, MPI_DOUBLE, step, tmprecvcount, tmpdisplace, MPI_DOUBLE, MPI_COMM_WORLD);
					communication += global_index_length / global_n;
					fnew = fun_obj->f_update(w, step, Q, eta, &index[curr_idx][0], global_index_length, tmpdisplace[rank], tmprecvcount[rank]);
				}
				else
				{
					if (index_length < length) // Change from sparse storage to dense storage for the local update step
						for (i=index_length-1;i >= 0; i--)
						{
							if (local_index[i] == i)
								break;
							local_step[local_index[i]] = local_step[i];
							local_step[i] = 0;
						}
					MPI_Allgatherv(local_step, length, MPI_DOUBLE, step, recv_count, displace, MPI_DOUBLE, MPI_COMM_WORLD);
					communication += 1.0;
					fnew = fun_obj->f_update(w, step, Q, eta);
				}
				alpha *= 2;
				counter++;
			} while (counter < max_modify_Q && (Q > 0 || (fnew - f > eta * Q)));

			if (counter < max_modify_Q && Q < 0)
			{
				f = fnew;
				if (sparse_communication_flag) // sparse communication
				{
					for (i=0;i<global_index_length;i++)
						w[index[curr_idx][i]] += step[i];
					for (i=0;i<index_length;i++)
						tmps[local_index[i]] = local_step[i];
				}
				else
				{
					daxpy_(&n, &one, step, &inc, w, &inc);
					memcpy(tmps, local_step, sizeof(double) * length);
				}
				inner_iter++;
				total_iter++;
			}
			else
				Q = 0;
		}
	}

	delete[] step;
	delete[] tmprecvcount;
	delete[] tmpdisplace;
	delete[] s;
	delete[] tmpy;
	delete[] tmps;
	delete[] loss_g;
	delete[] local_step;
	for (i=0; i < M; i++)
		delete[] inner_product_matrix[i];
	delete[] inner_product_matrix;
	delete[] R;
	delete[] full_g;
}

double MADPQN::Q_prox_grad(double *step, double *w, double *g, int global_length, int local_length, int local_start, int *index, double alpha, double *tmps, double *f, int *counter_ret)
{
	int inc = 1;
	double minus_one = -1.0;
	int counter = 0;
	double *wptr;
	double *gptr;
	const int max_modify_Q = 30;
	int w_size = fun_obj->get_nr_variable();
	int i;
	double Q = 0;
	double fnew = *f;
	if (global_length < w_size)
	{
		wptr = new double[global_length];
		gptr = new double[global_length];
		for (i=0; i<global_length;i++)
		{
			wptr[i] = w[index[i]];
			gptr[i] = g[index[i]];
		}
	}
	else
	{
		wptr = w;
		gptr = g;
	}

	do
	{
		memset(step, 0, sizeof(double) * global_length);
		fun_obj->prox_grad(wptr, gptr, step, step, alpha, global_length);
		Q = fun_obj->regularizer(step + local_start, local_length) - fun_obj->regularizer(wptr + local_start, local_length);
		daxpy_(&global_length, &minus_one, wptr , &inc, step, &inc);
		Q += 0.5 * alpha * ddot_(&local_length, step + local_start, &inc, step + local_start, &inc) + ddot_(&local_length, gptr + local_start, &inc, step + local_start, &inc);
		mpi_allreduce(&Q, 1, MPI_DOUBLE, MPI_SUM);
		communication += 1 / global_n;
		fnew = fun_obj->f_update(w, step, Q, eta, index, global_length, local_start, local_length);
		alpha *= 2;
		counter++;
	} while (counter < max_modify_Q && (Q > 0 || (fnew - (*f) > eta * Q)));
	*counter_ret = counter;

	if (counter < max_modify_Q && Q < 0)
	{
		*f = fnew;
		for (i=0;i<global_length;i++)
			w[index[i]] += step[i];
		for (i=0;i<global_length;i++)
		{
			int idx = index[i];
			if (idx >= start + length)
				break;
			if (idx >= start)
				tmps[idx - start] = step[i];
		}
	}
	else
		Q = 0;
	if (global_length < w_size)
	{
		delete[] wptr;
		delete[] gptr;
	}
	return Q;
}

void MADPQN::update_inner_products(double **inner_product_matrix, int k, int DynamicM, double *s, double *y, const std::vector<int> &index)
{
	int i;
	int inc = 1;
	char T[] = "T";
	double zero = 0;
	double one = 1.0;

	double *buffer = new double[DynamicM * 2];
	if (index[0] == -1 && length > 0)
	{
		dgemv_(T, &length, &DynamicM, &one, s, &length, s + k * length, &inc, &zero, buffer, &inc);
		dgemv_(T, &length, &DynamicM, &one, y, &length, s + k * length, &inc, &zero, buffer + DynamicM, &inc);
	}
	else
	{
		double *current_s = s + k * length;
		for (i=0;i<DynamicM;i++)
		{
			buffer[i] = 0;
			double *si = s + i * length;
			for (std::vector<int>::const_iterator it = index.begin(); it != index.end(); it++)
				buffer[i] += current_s[*it] * si[*it];
		}
		for (i=0;i<DynamicM;i++)
		{
			buffer[DynamicM + i] = 0;
			double *yi = y + i * length;
			for (std::vector<int>::const_iterator it = index.begin(); it != index.end(); it++)
				buffer[DynamicM + i] += current_s[*it] * yi[*it];
		}
	}

	mpi_allreduce(buffer, 2 * DynamicM, MPI_DOUBLE, MPI_SUM);
	communication += 2 * DynamicM / global_n;
	for (i=0;i<DynamicM;i++)
	{
		inner_product_matrix[k][2 * i] = buffer[i];
		inner_product_matrix[i][2 * k] = buffer[i];
		inner_product_matrix[k][2 * i + 1] = buffer[DynamicM + i];
	}
	delete[] buffer;
}

void MADPQN::compute_R(double *R, int DynamicM, double **inner_product_matrix, int k, double gamma)
{
	int i,j;
	int size = 2 * DynamicM;
	int sizesquare = size * size;

	memset(R, 0, sizeof(double) * sizesquare);

	//R is a symmetric matrix
	//(1,1) block, S^T S
	for (i=0;i<DynamicM;i++)
		for (j=0;j<DynamicM;j++)
			R[i * size + j] = gamma * inner_product_matrix[i][2 * j];

	//(2,2) block, D = diag(s_i^T y_i)
	for (i=0;i<DynamicM;i++)
		R[(DynamicM + i) * (size + 1)] = -inner_product_matrix[i][2 * i + 1];

	//(1,2) block, L = tril(S^T Y, -1), and (2,1) block, L^T
	for (i=1;i<DynamicM;i++)
	{
		int idx = (k + 1 + i) % DynamicM;
		for (j=0;j<i;j++)
		{
			int idxj = (k + 1 + j) % DynamicM;
			R[(DynamicM + idxj) * size + idx] = inner_product_matrix[idx][2 * idxj + 1];
			R[idx * size + DynamicM + idxj] = inner_product_matrix[idx][2 * idxj + 1];
		}
	}

	inverse(R, size);
}

double MADPQN::SpaRSA(double *w, double *loss_g, double *R, double *s, double *y, double gamma, double *local_step, double scaling, int DynamicM, std::vector<int> &index)
{
	const double eta = .01 / 2;
	const double ALPHA_MAX = 1e10;
	const double ALPHA_MIN = 1e-4;

	int i,j;
	double one = 1.0;
	int inc = 1;
	double minus_one = -one;
	double zero = 0;
	double dnorm0 = 0;
	double dnorm;
	char N[] = "N";
	char T[] = "T";
	//parameters for blas
	//Scaling is for the quadratic part

	int iter = 0;
	int Rsize = 2 * DynamicM;

	double *oldd = new double[index.size()];
	double *oldg_subprob = new double[index.size()];
	double *g_subprob = new double[index.size()]; //gradient of the quadratic subproblem
	double *ddiff = new double[index.size()];//ddiff = oldd - newd
	double *SYTd = new double[Rsize + 3];
	double *tmp = new double[Rsize];
	double *subw = NULL;//The part of w corresponds to the index
	double *sub_loss_g = NULL;
	double *subs = NULL;
	double *suby = NULL;
	double oldreg = 0;
	double reg = 0;

	double oldquadratic = 0;
	double alpha = 0;
	double all_reduce_buffer;
	int proxgradcounts = 0;
	double accumulated_improve = 0;

	if (index.size() == 0)
	{
		while (iter < max_inner)
		{
			if (iter != 0)
			{
				all_reduce_buffer = 0;
				mpi_allreduce(&all_reduce_buffer, 1, MPI_DOUBLE, MPI_SUM);
				communication += 1 / global_n;
				alpha = all_reduce_buffer / SYTd[Rsize + 2];
			}
			else
				alpha = gamma * scaling;

			alpha = max(ALPHA_MIN, alpha);
			double fun_improve = 0.0;
			double rhs = 0.0;
			double tmpquadratic = 0;
			int times = 0;

			//Line search stopping: f^+ - f < -eta * alpha * |x^+ - x|^2
			while (fun_improve >= rhs)
			{
				times++;
				if (alpha > ALPHA_MAX)
					break;
				proxgradcounts++;

				memset(SYTd, 0, (Rsize + 3) * sizeof(double));
				mpi_allreduce(SYTd, Rsize + 3, MPI_DOUBLE, MPI_SUM);
				communication += (Rsize + 3) / global_n;

				// tmp = R ([gamma S Y]^T d)
				for (i=0;i<Rsize;i++)
				{
					tmp[i] = 0;
					for (j=0;j<Rsize;j++)
						tmp[i] += R[i * Rsize + j] * SYTd[j];
				}
				//			dsymv_(U, &Rsize, &one, R, &Rsize, SYTd, &inc, &zero, tmp, &inc);

				tmpquadratic = scaling * (gamma * SYTd[Rsize] - ddot_(&Rsize, SYTd, &inc, tmp, &inc));
				fun_improve = SYTd[Rsize + 1] + (tmpquadratic - oldquadratic) / 2; //SYTd[Rsize + 1] records the change of the regularizer
				rhs = -eta * alpha * SYTd[Rsize + 2]; // SYTd[Rsize + 2] = d^T d

				alpha *= 2.0;
			}
			dnorm = sqrt(SYTd[Rsize + 2]);
			if (iter == 0)
				dnorm0 = dnorm;
			oldquadratic = tmpquadratic;
			accumulated_improve += SYTd[Rsize + 1];

			if (alpha > ALPHA_MAX)
				break;
			if (iter > 0 && dnorm / dnorm0 < inner_eps)
				break;
			iter++;
		}
	}
	else
	{
		if ((int) index.size() == length)
		{
			subs = s;
			suby = y;
			sub_loss_g = loss_g;
			subw = w;
		}
		else
		{
			subs = new double[index.size() * DynamicM * 2];
			suby = subs + (index.size() * DynamicM);
			subw = new double[index.size()];
			sub_loss_g = new double[index.size()];
			for (i=0;i<(int)index.size();i++)
			{
				sub_loss_g[i] = loss_g[index[i]];
				subw[i] = w[index[i]];
			}

			// Take all the vectors according to the indices first so that we can use blas
			for (i=0;i<DynamicM;i++)
			{
				double *tmpsubs = subs + (index.size() * i);
				double *tmps = s + (length * i);
				for (j=0;j<(int)index.size();j++)
					tmpsubs[j] = tmps[index[j]];
			}
			for (i=0;i<DynamicM;i++)
			{
				double *tmpsuby = suby + (index.size() * i);
				double *tmpy = y + (length * i);
				for (j=0;j<(int)index.size();j++)
					tmpsuby[j] = tmpy[index[j]];
			}
		}

		memset(oldd, 0, sizeof(double) * index.size());
		memcpy(oldg_subprob, sub_loss_g, index.size() * sizeof(double));
		memset(local_step, 0, sizeof(double) * index.size());
		memset(SYTd, 0, (Rsize + 3) * sizeof(double));
		double factor = scaling * gamma;
		double minus_gamma_scaling = -factor;
		double minus_scaling = -scaling;

		int index_length = (int) index.size();
		oldreg = fun_obj->regularizer(subw, index_length);

		while (iter < max_inner)
		{
			memcpy(g_subprob, sub_loss_g, index.size() * sizeof(double));
			if (iter != 0)
			{
				//compute the gradient of the sub-problem: g + scaling * (gamma * d - Q R Q^T d), with Q = [gamma * S, Y]
				//Note that RQ^Td is already obtained from the previous round in computing obj value
				//This vector is stored in tmp
				daxpy_(&index_length, &factor, local_step, &inc, g_subprob, &inc);
				dgemv_(N, &index_length, &DynamicM, &minus_gamma_scaling, subs, &index_length, tmp, &inc, &one, g_subprob, &inc);
				dgemv_(N, &index_length, &DynamicM, &minus_scaling, suby, &index_length, tmp + DynamicM, &inc, &one, g_subprob, &inc);

				//Now grad is ready, compute alpha = y^T s / s^T s, of the subprob
				daxpy_(&index_length, &minus_one, g_subprob, &inc, oldg_subprob, &inc);//get -y of the subprob

				all_reduce_buffer = ddot_(&index_length, ddiff, &inc, oldg_subprob, &inc);//Now oldg is actually oldg - g
				mpi_allreduce(&all_reduce_buffer, 1, MPI_DOUBLE, MPI_SUM);
				communication += 1 / global_n;
				alpha = all_reduce_buffer / SYTd[Rsize + 2];

				memcpy(oldg_subprob, g_subprob, sizeof(double) * index.size());
			}
			else
				alpha = factor;

			alpha = max(ALPHA_MIN, alpha);
			double fun_improve = 0.0;
			double rhs = 0.0;
			double tmpquadratic = 0;
			int times = 0;
			memset(SYTd, 0, (Rsize + 3) * sizeof(double));

			//Line search stopping: f^+ - f < -eta * alpha * |x^+ - x|^2
			while (fun_improve >= rhs)
			{
				times++;
				if (alpha > ALPHA_MAX)
					break;
				fun_obj->prox_grad(subw, g_subprob, local_step, oldd, alpha, (int)index.size());
				reg = fun_obj->regularizer(local_step, index_length);
				proxgradcounts++;
				daxpy_(&index_length, &minus_one, subw, &inc, local_step, &inc);
				memcpy(ddiff, oldd, sizeof(double) * index.size());
				daxpy_(&index_length, &minus_one, local_step, &inc, ddiff, &inc);

				//SYTd[Rsize + 2] records ||x^+ - x||^2
				SYTd[Rsize + 2] = ddot_(&index_length, ddiff, &inc, ddiff, &inc);

				//SYTd[Rsize + 1] records func diff in the terms linear to d: regularizer and g^T d
				SYTd[Rsize + 1] = -ddot_(&index_length, sub_loss_g, &inc, ddiff, &inc) + reg - oldreg;
				//SYTd[Rsize] records d^T d
				SYTd[Rsize] = ddot_(&index_length, local_step, &inc, local_step, &inc);

				dgemv_(T, &index_length, &DynamicM, &gamma, subs, &index_length, local_step, &inc, &zero, SYTd, &inc);
				dgemv_(T, &index_length, &DynamicM, &one, suby, &index_length, local_step, &inc, &zero, SYTd + DynamicM, &inc);

				mpi_allreduce(SYTd, Rsize + 3, MPI_DOUBLE, MPI_SUM);
				communication += (Rsize + 3) / global_n;

				for (i=0;i<Rsize;i++)
				{
					tmp[i] = 0;
					for (j=0;j<Rsize;j++)
						tmp[i] += R[i * Rsize + j] * SYTd[j];
				}
				tmpquadratic = scaling * (gamma * SYTd[Rsize] - ddot_(&Rsize, SYTd, &inc, tmp, &inc));
				fun_improve = SYTd[Rsize + 1] + (tmpquadratic - oldquadratic) / 2;
				rhs = -eta * alpha * SYTd[Rsize + 2];

				alpha *= 2.0;
			}
			oldreg = reg;
			dnorm = sqrt(SYTd[Rsize + 2]);
			if (iter == 0)
				dnorm0 = dnorm;
			oldquadratic = tmpquadratic;
			accumulated_improve += SYTd[Rsize + 1];

			if (alpha > ALPHA_MAX)
			{
				memcpy(local_step, oldd, sizeof(double) * index.size());
				break;
			}
			memcpy(oldd, local_step, sizeof(double) * index.size());

			if (iter > 0 && dnorm / dnorm0 < inner_eps)
				break;
			iter++;
		}
	}
	if (iter == 1)
		inner_eps /= 4.0;
	delete[] oldd;
	delete[] oldg_subprob;
	delete[] g_subprob;
	delete[] ddiff;
	delete[] SYTd;
	delete[] tmp;
	if (index.size() != 0 && (int) index.size() != length)
	{
		delete[] subw;
		delete[] sub_loss_g;
		delete[] subs;
	}
	return oldquadratic / 2 + accumulated_improve;
}


double MADPQN::newton(double *g, double *step, const std::vector<int> &global_nonzero_set, int max_cg_iter, double *w)
{
	int sub_length = (int) global_nonzero_set.size();
	const double psd_threshold = 1e-8;
	double eps_cg = 0.1;

	int i, inc = 1;
	double one = 1;
	double *d = new double[sub_length];
	double *Hd = new double[sub_length];
	double zTr, znewTrnew, alpha, beta, cgtol;
	double *z = new double[sub_length];
	double *r = new double[sub_length];
	double *M = new double[sub_length];
	double sTs = 0, sHs = 0, dHd = 0, dTd = 0;

	double alpha_pcg = 1.0;
	fun_obj->get_diag_preconditioner(M, global_nonzero_set, w);
	for(i=0; i<sub_length; i++)
		M[i] = (1-alpha_pcg) + alpha_pcg*M[i];

	for (i=0; i<sub_length; i++)
	{
		step[i] = 0.0;
		r[i] = -g[i];
		z[i] = r[i] / M[i];
		d[i] = z[i];
	}
	zTr = ddot_(&sub_length, z, &inc, r, &inc);
	double rTr = ddot_(&sub_length, r, &inc, r, &inc);
	double gs = 0.0;

	cgtol = eps_cg * min(1.0, rTr);
	double cg_boundary = 1e+6 * ddot_(&sub_length, z, &inc, z, &inc);
	int cg_iter = 0;

	while (cg_iter < max_cg_iter)
	{
		if (sqrt(rTr) <= cgtol)
			break;
		cg_iter++;
		fun_obj->Hv(d, Hd, global_nonzero_set, w);
		dHd = ddot_(&sub_length, d, &inc, Hd, &inc);
		dTd = ddot_(&sub_length, d, &inc, d, &inc);
		if (dHd / dTd <= psd_threshold)
		{
			info("WARNING: dHd / dTd <= PSD threshold\n");
			break;
		}

		alpha = zTr/dHd;
		daxpy_(&sub_length, &alpha, d, &inc, step, &inc);
		alpha = -alpha;
		gs = ddot_(&sub_length, g, &inc, step, &inc);
		if (gs >= 0)
		{
			info("gs >= 0 in CG\n");
			daxpy_(&sub_length, &alpha, d, &inc, step, &inc);
			break;
		}

		sTs = ddot_(&sub_length, step, &inc, step, &inc);
		if (sTs >= cg_boundary)
		{
			info("WARNING: reaching cg boundary\n");
			break;
		}

		daxpy_(&sub_length, &alpha, Hd, &inc, r, &inc);
		sHs = -ddot_(&sub_length, r, &inc, step, &inc) - gs;
		if (sHs / sTs <= psd_threshold)
		{
			info("WARNING: sHs / sTs <= PSD threshold\n");
			break;
		}

		for (i=0; i<sub_length; i++)
			z[i] = r[i] / M[i];
		znewTrnew = ddot_(&sub_length, z, &inc, r, &inc);

		beta = znewTrnew/zTr;
		dscal_(&sub_length, &beta, d, &inc);
		daxpy_(&sub_length, &one, z, &inc, d, &inc);
		zTr = znewTrnew;
		rTr= ddot_(&sub_length, r, &inc, r, &inc);
	}
	double delta =  0.5 * sHs + gs;
	if (gs < 0 && delta >= 0)
		delta = gs;

	info("CG Iter = %d\n", cg_iter);

	if (cg_iter == max_cg_iter)
		info("WARNING: reaching maximal number of CG steps\n");

	delete[] d;
	delete[] Hd;
	delete[] z;
	delete[] r;
	delete[] M;

	return delta;
}

// end of solvers

static void transpose(const problem *prob, feature_node **x_space_ret, problem *prob_col)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	size_t nnz = 0;
	size_t *col_ptr = new size_t [n+1];
	feature_node *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = new double[l];
	prob_col->x = new feature_node*[n];

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i+1; // starts from 1
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	delete [] col_ptr;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int i;

	std::set<int> label_set;
	for(i=0;i<prob->l;i++)
		label_set.insert((int)prob->y[i]);
	
	int label_size = (int)label_set.size();
	int num_machines = mpi_get_size();
	int max_label_size;
	MPI_Allreduce(&label_size, &max_label_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

	std::vector<int> global_label_sets((max_label_size+1)*num_machines);
	std::vector<int> label_buff(max_label_size+1);

	label_buff[0] = label_size;
	i = 1;
	for(std::set<int>::iterator this_label=label_set.begin();
			this_label!=label_set.end(); this_label++)
	{
		label_buff[i] = (*this_label);
		i++;
	}
	
	MPI_Allgather(label_buff.data(), max_label_size+1, MPI_INT, global_label_sets.data(), max_label_size+1, MPI_INT, MPI_COMM_WORLD);

	for(i=0; i<num_machines; i++)
	{
		int offset = i*(max_label_size+1);
		int size = global_label_sets[offset];
		for(int j=1; j<=size; j++)
			label_set.insert(global_label_sets[offset+j]);
	}

	int nr_class = (int)label_set.size();

	std::map<int, int> label_map;
	int *label = Malloc(int, nr_class);
	i = 0;
	for(std::set<int>::iterator this_label=label_set.begin();
			this_label!=label_set.end(); this_label++)
	{
		label[i] = (*this_label);
		i++;
	}
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
		std::swap(label[0], label[1]);
	for(i=0;i<nr_class;i++)
		label_map[label[i]] = i;


	// The following codes are similar to the original LIBLINEAR
	int *start = Malloc(int, nr_class);
	int *count = Malloc(int, nr_class);
	for(i=0;i<nr_class;i++)
		count[i] = 0;
	for(i=0;i<prob->l;i++)
		count[label_map[(int)prob->y[i]]]++;

	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[label_map[(int)prob->y[i]]]] = i;
		++start[label_map[(int)prob->y[i]]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
}

static void train_one(const problem *prob, const parameter *param, double *w)
{
	double eps = param->eps;

	int l = prob->l;
	int pos = 0;
	int neg = 0;
	for(int i=0;i<l;i++)
		if(prob->y[i] > 0)
			pos++;
	mpi_allreduce(&pos, 1, MPI_INT, MPI_SUM);
	mpi_allreduce(&l, 1, MPI_INT, MPI_SUM);
	neg = l - pos;

	double primal_solver_tol = (eps*max(min(pos,neg), 1))/l;
	if (param->problem_type == LASSO)
		primal_solver_tol = eps;

	l1r_fun *l1r_fun_obj=NULL;
	problem prob_col;
	feature_node *x_space = NULL;
	transpose(prob, &x_space ,&prob_col);

	int n = prob->n;
	double *w_tmp = NULL;
	feature_node **x = NULL;
	std::vector<int> order;

	if (param->permute_features)
	{
		fill_range(order, n);
		srand(1);
		std::random_shuffle(order.begin(), order.end());

		x = new feature_node*[n];
		for(int i = 0; i < n; i++)
			x[i] = prob_col.x[i];
		for(int i = 0; i < n; i++)
			prob_col.x[i] = x[order[i]];
	}

	switch(param->problem_type)
	{
		case L1R_LR:
			l1r_fun_obj = new l1r_lr_fun(&prob_col, param->C);
			break;
		case LASSO:
			l1r_fun_obj = new lasso(&prob_col, param->C);
			break;
		case L1R_L2_LOSS_SVC:
			l1r_fun_obj = new l1r_l2_svc_fun(&prob_col, param->C);
			break;
		default:
			if(mpi_get_rank() == 0)
				fprintf(stderr, "ERROR: unknown problem_type\n");
	}

	MADPQN madpqn_obj(l1r_fun_obj, primal_solver_tol, param->m, param->inner_eps, param->max_inner_iter, param->eta);
	madpqn_obj.set_print_string(liblinear_print_string);
	madpqn_obj.madpqn(w, param->disable_smooth);

	if (param->permute_features)
	{
		w_tmp = new double[n];
		for(int i = 0; i < n; i++)
			w_tmp[i] = w[i];
		for(int i = 0; i < n; i++)
			w[order[i]] = w_tmp[i];
	}

	delete l1r_fun_obj;
	delete [] prob_col.y;
	delete [] prob_col.x;
	delete [] x_space;
	if (param->permute_features)
	{
		delete [] x;
		delete [] w_tmp;
	}
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	if(check_regression_model(model_))
	{
		model_->w = Malloc(double, w_size);
		for(i=0; i<w_size; i++)
			model_->w[i] = 0;
		model_->nr_class = 2;
		model_->label = NULL;
		train_one(prob, param, model_->w);
	}
	else
	{
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		group_classes(prob,&nr_class,&label,&start,&count,perm);

		model_->nr_class=nr_class;
		model_->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model_->label[i] = label[i];

		// constructing the subproblem
		feature_node **x = Malloc(feature_node *,l);
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		int k;
		problem sub_prob;
		sub_prob.l = l;
		sub_prob.n = n;
		sub_prob.x = Malloc(feature_node *,sub_prob.l);
		sub_prob.y = Malloc(double,sub_prob.l);

		for(k=0; k<sub_prob.l; k++)
			sub_prob.x[k] = x[k];

		if(param->problem_type == GROUPLASSO_MLR)
		{
			model_->w=Malloc(double, n*nr_class);
			memset(model_->w, 0, sizeof(double) * n * nr_class);
			for(i=0;i<nr_class;i++)
				for(j=start[i];j<start[i]+count[i];j++)
					sub_prob.y[j] = i;
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(&sub_prob, &x_space ,&prob_col);

			int n = prob->n;
			double *w_tmp = NULL;
			feature_node **x = NULL;
			std::vector<int> order;

			if (param->permute_features)
			{
				fill_range(order, n);
				srand(1);
				std::random_shuffle(order.begin(), order.end());

				x = new feature_node*[n];
				for(int i = 0; i < n; i++)
					x[i] = prob_col.x[i];
				for(int i = 0; i < n; i++)
					prob_col.x[i] = x[order[i]];
			}
			grouplasso_mlr_fun mlr_fun_obj(&prob_col, param->C, nr_class);

			MADPQN madpqn_obj(&mlr_fun_obj, param->eps, param->m, param->inner_eps, param->max_inner_iter, param->eta);
			madpqn_obj.set_print_string(liblinear_print_string);
			madpqn_obj.madpqn(model_->w, param->disable_smooth);

			if (param->permute_features)
			{
				w_tmp = new double[n * nr_class];
				for(int i = 0; i < n * nr_class; i++)
					w_tmp[i] = model_->w[i];
				for(int i = 0; i < n; i++)
					for (int j = 0; j < nr_class; j++)
						model_->w[order[i] * nr_class + j] = w_tmp[i * nr_class + j];
			}

			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			if (param->permute_features)
			{
				delete [] x;
				delete [] w_tmp;
			}
		}
		else
		{
			if(nr_class == 2)
			{
				model_->w=Malloc(double, w_size);

				int e0 = start[0]+count[0];
				k=0;
				for(; k<e0; k++)
					sub_prob.y[k] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;

				for(i=0;i<w_size;i++)
					model_->w[i] = 0;

				train_one(&sub_prob, param, model_->w);
			}
			else
			{
				model_->w=Malloc(double, w_size*nr_class);
				double *w=Malloc(double, w_size);
				for(i=0;i<nr_class;i++)
				{
					int si = start[i];
					int ei = si+count[i];

					k=0;
					for(; k<si; k++)
						sub_prob.y[k] = -1;
					for(; k<ei; k++)
						sub_prob.y[k] = +1;
					for(; k<sub_prob.l; k++)
						sub_prob.y[k] = -1;

					for(j=0;j<w_size;j++)
						w[j] = 0;

					train_one(&sub_prob, param, w);

					for(j=0;j<w_size;j++)
						model_->w[j*nr_class+i] = w[j];
				}
				free(w);
			}
		}

		free(x);
		free(label);
		free(start);
		free(count);
		free(perm);
		free(sub_prob.x);
		free(sub_prob.y);
	}
	return model_;
}


double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2)
	{
		if(check_regression_model(model_))
			return dec_values[0];
		else
			return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	}
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(check_probability_model(model_))
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		double label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;
	}
	else
		return 0;
}

static const char *problem_type_table[]=
{
	"L1R_LR", "LASSO","L1R_L2_LOSS_SVC","MLR",NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2 && param.problem_type != GROUPLASSO_MLR)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "problem_type %s\n", problem_type_table[param.problem_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.17g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.17g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);


	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var)do\
{\
	if (fscanf(_stream, _format, _var) != 1)\
	{\
		fprintf(stderr, "ERROR: fscanf failed to read the model\n");\
		EXIT_LOAD_MODEL()\
	}\
}while(0)
// EXIT_LOAD_MODEL should NOT end with a semicolon.
#define EXIT_LOAD_MODEL()\
{\
	free(model_->label);\
	free(model_);\
	return NULL;\
}
struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);
		if(strcmp(cmd,"problem_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;problem_type_table[i];i++)
			{
				if(strcmp(problem_type_table[i],cmd)==0)
				{
					param.problem_type=i;
					break;
				}
			}
			if(problem_type_table[i] == NULL)
			{
				fprintf(stderr,"[rank %d] unknown solver type.\n", mpi_get_rank());
				EXIT_LOAD_MODEL()
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			FSCANF(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			FSCANF(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			FSCANF(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				FSCANF(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"[rank %d] unknown text in model file: [%s]\n",mpi_get_rank(),cmd);
			EXIT_LOAD_MODEL()
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			FSCANF(fp, "%lf ", &model_->w[i*nr_w+j]);
	}


	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

// use inline here for better performance (around 20% faster than the non-inline one)
static inline double get_w_value(const struct model *model_, int idx, int label_idx)
{
	int nr_class = model_->nr_class;
	const double *w = model_->w;

	if(idx < 0 || idx > model_->nr_feature)
		return 0;
	if(label_idx < 0 || label_idx >= nr_class)
		return 0;
	if(nr_class == 2)
		if(label_idx == 0)
			return w[idx];
		else
			return -w[idx];
	else
		return w[idx*nr_class+label_idx];
}

// feat_idx: starting from 1 to nr_feature
// label_idx: starting from 0 to nr_class-1 for classification models;
//            for regression models, label_idx is ignored.

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->problem_type != L1R_LR
	&& param->problem_type != LASSO
	&& param->problem_type != L1R_L2_LOSS_SVC
	&& param->problem_type != GROUPLASSO_MLR)
		return "unknown problem type";

	return NULL;
}

int check_probability_model(const struct model *model_)
{
	return (model_->param.problem_type==L1R_LR);
}

int check_regression_model(const struct model *model_)
{
	return (model_->param.problem_type==LASSO);
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

int mpi_get_rank()
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank;	
}

int mpi_get_size()
{
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;	
}

void mpi_exit(const int status)
{
	MPI_Finalize();
	exit(status);
}
