#include "lum_retinex.h"

#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <chrono>

namespace lum {

    // timing helper
	double get_s() {
		using namespace std::chrono;
		auto now = system_clock::now();
		system_clock::duration tse = now.time_since_epoch();
		return duration_cast<nanoseconds>(tse).count() / 1e9;
	}

    // Helper image processing routines
	void log(const float* inimg, float* outimg, int sz) {
#ifdef USE_MKL
		vsLn(sz, inimg, outimg);
#else
		for (int i = 0; i < sz; ++i) {
			float in = inimg[i];
			outimg[i] = std::logf(inimg[i] + 0.00001);
		}
#endif
	}

	void exp(const float* inimg, float* outimg, int sz) {
#ifdef USE_MKL
		vsExp(sz, inimg, outimg);
#else
		for (int i = 0; i < sz; ++i) {
			outimg[i] = std::expf(inimg[i]);
		}
#endif
	}
	void mean(const float* inimg, float* outimg, int outsz) {
		for (int i = 0; i < outsz; ++i) {
			outimg[i] = (inimg[i * 3] + inimg[i * 3 + 1] + inimg[i * 3 + 2]) / 3;
		}
	}


	// helpers for image indices
	class reflshadidx {
	public:
		reflshadidx(int w, int h)
		:m_w(w), m_h(h){}
		int reflidx(int x, int y) const {
			return m_w * y + x;
		}
		int shadidx(int x, int y) const {
			return m_w * m_h + reflidx(x, y);
		}
	private:
		int m_w;
		int m_h;
	};
	class imwrap {
	public:
		imwrap(const float* img, int w, int h)
		:m_w(w), m_h(h), m_img(img){ }

		float operator()(int x, int y) const {
			assert(x >= 0);
			assert(y >= 0);
			assert(x < m_w);
			assert(y < m_h);
			return m_img[m_w * y + x];
		}
	private:
		const float* m_img;
		int m_w;
		int m_h;
	};

	// forward declaration of internal functions
	void reflect_clamp(int w, int h, float* refl_in, float* shading_in);
	void preprocess(int w, int h, const float* img, float* logimg);
	void postprocess(int w, int h, float* refl_in, float* shading_in, float* refl_out, float* shading_out);
	Eigen::VectorXf makeB(float threshold, const float* im, int w, int h);

	using Triplet = Eigen::Triplet<float>;
	int nconstraints(int w, int h) {
		return w*h + 2 * w*(h - 1) + 2 * (w - 1) * h;
	}
	int nentries(int w, int h) {
		return 2 * (w*h + 2 * w*(h - 1) + 2 * (w - 1) * h);
	}
	std::vector<Triplet> makeTriplets(int w, int h) {
		reflshadidx I(w, h);
		printf("Assemble matrix.\n");
		double assemble_start = get_s();
		std::vector <Triplet> entries;
		int cit = 0;
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				if (x < w - 1) {
					// dxR(r, c) = -lR(r, c) + lR(r, c + 1)
					entries.push_back(Triplet(cit, I.reflidx(x, y), -1));
					entries.push_back(Triplet(cit, I.reflidx(x + 1, y), +1));
					cit++;

					// dxS(r, c) = -lS(r, c) + lS(r, c + 1)
					entries.push_back(Triplet(cit, I.shadidx(x, y), -1));
					entries.push_back(Triplet(cit, I.shadidx(x + 1, y), +1));
					cit++;
				}
				if (y < h - 1) {
					entries.push_back(Triplet(cit, I.reflidx(x, y), -1));
					entries.push_back(Triplet(cit, I.reflidx(x, y + 1), +1));
					cit++;

					entries.push_back(Triplet(cit, I.shadidx(x, y), -1));
					entries.push_back(Triplet(cit, I.shadidx(x, y + 1), +1));
					cit++;
					// dyR(r, c) = -lR(r, c) + lR(r + 1)
					// dyS(r, c) = -lS(r, c) + lS(r + 1)
				}

				// reflectance plus shading (log space) == final image
				entries.push_back(Triplet(cit, I.reflidx(x, y), 1));
				entries.push_back(Triplet(cit, I.shadidx(x, y), 1));
				cit++;
			}
		}
		assert(entries.size() == nentries(w, h));
		double assemble_end = get_s();
		printf("Makemtx took %.1fms\n", (assemble_end- assemble_start) * 1000);
		return entries;
	}

	// to log domain
	void preprocess(int w, int h, const float* img, float* logimg) {
		printf("Start Preprocessing.\n");
		double preprocess_start = get_s();
		log(img, logimg, w*h);
		double preprocess_b_end = get_s();
		printf("Preprocess took %.1fms\n", (preprocess_b_end - preprocess_start) * 1000);
	}

	// back to linear
	void postprocess(int w, int h, float* refl_in, float* shading_in, float* refl_out, float* shading_out) {
		printf("Post process start.\n");
		double postprocess_start = get_s();
		reflect_clamp(w, h, refl_in, shading_in);
		exp(refl_in, refl_out, w*h);
		exp(shading_in, shading_out, w*h);
		double postprocess_end = get_s();
		printf("Postprocess took %.1fms\n", (postprocess_end - postprocess_start)*1000 );
	}

	Eigen::VectorXf makeB(float threshold, const float* im, int w, int h) {
		Eigen::VectorXf b(nconstraints(w, h));
		double assemble_b_start = get_s();
		imwrap I(im, w, h);
		int cit = 0;
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				if (x < w - 1) {
					float dx = -I(x, y) + I(x + 1, y);
					float dxR;
					float dxS;
					if (std::abs(dx) > threshold) {
						dxR = dx;
						dxS = 0;
					} else {
						dxR = 0;
						dxS = dx;
					}
					// dxR(r, c) = -lR(r, c) + lR(r, c + 1)
					b(cit++) = dxR;

					// dxS(r, c) = -lS(r, c) + lS(r, c + 1)
					b(cit++) = dxS;
				}
				if (y < h - 1) {
					float dy = -I(x, y) + I(x, y + 1);
					float dyR;
					float dyS;
					if (std::abs(dy) > threshold) {
						dyR = dy;
						dyS = 0;
					} else {
						dyR = 0;
						dyS = dy;
					}
					b(cit++) = dyR;
					b(cit++) = dyS;
					// dyR(r, c) = -lR(r, c) + lR(r + 1)
					// dyS(r, c) = -lS(r, c) + lS(r + 1)
				}
				// reflectance plus shading (log space) == final image
				b(cit++) = I(x, y);
			}
		}
		return b;
	}

	// operates on log-reflectance
	// makes sure that log-reflectane is less than 0
	void reflect_clamp(int w, int h, float* refl_in, float* shading_in) {

		float max_reflectance = -FLT_MIN;
		float min_reflectance = FLT_MAX;
		int nancount = 0;
		int infcount = 0;
		for (int i = 0; i < w * h; ++i) {
			if (refl_in[i] > max_reflectance) {
				max_reflectance = refl_in[i];
			}
			if (refl_in[i] < min_reflectance) {
				min_reflectance = refl_in[i];
			}
		}

		if (max_reflectance > 0) {
			for (int i = 0; i < w * h; ++i) {
				refl_in[i] -= max_reflectance;
			}
			for (int i = 0; i < w * h; ++i) {
				shading_in[i] += max_reflectance;
			}
		}
	}

	void retinex(float threshold, const float* im, int w, int h, float* reflectance, float* shading) {
		assert(reflectance);
		assert(shading);
		assert(im);
		retinex_decomp rdec(w, h);
		rdec.solve(threshold, im, reflectance, shading);
	}
	/* I would prefer solving direction Ax = b.
	  However, this doesn't work with Eigen's QR decomp (why?).
	  I solve A'Ax = A'b, using Cholesky, instead.
	*/
	retinex_decomp::retinex_decomp(int w, int h): m_w(w), m_h(h) {
		std::vector <Triplet> entries = makeTriplets(w, h);
		SpMat A(nconstraints(w, h), w * h * 2);
		A.setFromTriplets(entries.begin(), entries.end());
		m_At = A.transpose();
		{
			printf("factorize %d-by-%d matrix\n", A.cols(), A.cols());
			double decompose_start = get_s();
			m_solver.compute(m_At * A);
			double decompose_end = get_s();
			printf("factorize took %.1fms\n", (decompose_end - decompose_start) * 1000);
		}
	}
	void retinex_decomp::solve(float threshold, const float* im, float* reflectance, float* shading) {
		assert(reflectance);
		assert(shading);
		assert(im);

		std::vector<float> logimg(m_w*m_h);
		preprocess(m_w, m_h, im, logimg.data());
		Eigen::VectorXf b = makeB(threshold, logimg.data(), m_w, m_h);
		double solve_start = get_s();
		Eigen::VectorXf x = m_solver.solve(m_At * b);
		double solve_end = get_s();
		printf("solve took %.1fms\n", (solve_end - solve_start) * 1000);
		postprocess(m_w, m_h, x.data(), x.data() + m_w * m_h, reflectance, shading);
	}

	// does greyscale retinex with post processing.
	void retinex_decomp::solve_rgb(float threshold, const float* im, float* reflectance, float* shading) {
		assert(reflectance);
		assert(shading);
		assert(im);
		const int sz = m_w * m_h;

		std::vector<float> grayimg(sz);
		mean(im, grayimg.data(), grayimg.size());
		std::vector<float> graylog(sz);
		double before = get_s();
		log(grayimg.data(), graylog.data(), sz);
		double after = get_s();

		Eigen::VectorXf b = makeB(threshold, graylog.data(), m_w, m_h);
		double solve_start = get_s();
		Eigen::VectorXf x = m_solver.solve(m_At * b);
		double solve_end = get_s();

		float* log_shading = x.data() + sz;
		reflect_clamp(m_w, m_h, x.data(), log_shading);

		std::vector<float> rgb_logimg(3*sz);
		log(im, rgb_logimg.data(), rgb_logimg.size());

		// do gray to rgb conversion
		// log R = log I - log S
		for (int i = 0; i < sz; ++i) {
			rgb_logimg[i * 3 + 0] = rgb_logimg[i * 3 + 0] - log_shading[i];
			rgb_logimg[i * 3 + 1] = rgb_logimg[i * 3 + 1] - log_shading[i];
			rgb_logimg[i * 3 + 2] = rgb_logimg[i * 3 + 2] - log_shading[i];
		}

		exp(rgb_logimg.data(), reflectance, 3 * sz);
		exp(log_shading, shading, sz);
	}
}