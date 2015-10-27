#ifndef LUM_RETINEX_H
#define LUM_RETINEX_H

#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

namespace lum {

    // Run retinex as single function call (cannot reuse decomposition).
	void retinex(float threshold, const float* img, int w, int h, float* reflectance, float* shading);

	// First create decomposition for linear solver (slow)
	// Then solve for various right-hand-sides (fast)
	class retinex_decomp {
	public:
		retinex_decomp(int w, int h);
		void solve(float threshold, const float* img, float* refl, float* shading);
		void solve_rgb(float threshold, const float* img, float* refl, float* shading);
	private:
		int m_w;
		int m_h;
		using SpMat = Eigen::SparseMatrix < float > ;
		SpMat m_At;
		Eigen::SimplicialCholesky<SpMat> m_solver;
	};
}
#endif