#include <iostream>
#include <cmath>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "interpolation.h"

using namespace cv;
using namespace std;

namespace IPL {

#define clip(x, a, b) x >= a ? (x <= b ? x : b) : a;

    /**
      ******************************************************************************
      * @author         : oswin
      * @brief          : my function to resize image
      ******************************************************************************
      */
    void resizeBilinear_kernel(const cv::Mat &src_, cv::Mat &dst_, cv::Size &dsize) {
        int w = src_.cols;
        int h = src_.rows;
        int w2 = dsize.width;
        int h2 = dsize.height;
        dst_.create(dsize, 0);

        uchar *src = src_.data;
        uchar *dst = dst_.data;

        float x_ratio = float(w) / w2;
        float y_ratio = float(h) / h2;

        int total = dst_.total();

        for (int i = 0; i < total; i++) {
            // resize后的坐标值
            int dy = i / w2;
            int dx = i % w2;
            // printf("dx: %d | dy: %d \n", dx, dy);

            // 投影到原图的坐标值
//            float fx = (dx + 0.5) * x_ratio - 0.5;
//            float fy = (dy + 0.5) * y_ratio - 0.5;

            //test opencv
            float fx = dx * x_ratio;
            float fy = dy * y_ratio;

            fx = clip(fx, 0, w - 1);
            fy = clip(fy, 0, h - 1);

            // w0 和 h0
            int ix = std::floor(fx);
            int iy = std::floor(fy);
            ix = clip(ix, 0, w - 2)
            iy = clip(iy, 0, h - 2)
//            std::cout<<"orimap: "<<ix<<" iy: "<<iy<<std::endl;

            float u = fy - iy;
            float v = fx - ix;

            float _1_us = (1.f - u);// 1 - u
            float _1_vs = (1.f - v);// 1 - v

            //INTER_NEAREST
//            std::array<float, 4> dis{u + v, u + _1_vs, _1_us + _1_vs, _1_us + v};//f00,f10,f11,f01
//            auto max_id = std::max_element(dis.begin(), dis.end()) - dis.begin();
//            float tab[4] = {(float) src[iy * w + ix],
//                            (float) src[iy * w + (ix + 1)],
//                            (float) src[(iy + 1) * w + (ix + 1)],
//                            (float) src[(iy + 1) * w + ix]};

//            dst[dy * w2 + dx] = tab[(int) max_id];

            //test opencv
            dst[dy * w2 + dx] = (float) (src[iy * w + ix]);

            //INTER_LINEAR
            /*dst[dy * w2 + dx] = (float) (src[iy * w + ix] * _1_us * _1_vs) +
                                (float) (src[(iy + 1) * w + ix] * u * _1_vs) +
                                (float) src[iy * w + (ix + 1)] * v * _1_us +
                                (float) src[(iy + 1) * w + (ix + 1)] * u * v;*/

        }
    }



    /**
      ******************************************************************************
      * @author         : oswin
      * @brief          : opencv source code to resize down image
      ******************************************************************************
      */

    class resizeNNInvoker :
            public ParallelLoopBody
    {
    public:
        resizeNNInvoker(const Mat& _src, Mat &_dst, int *_x_ofs, double _ify) :
                ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs),
                ify(_ify)
        {
        }

        virtual void operator() (const Range& range) const CV_OVERRIDE
        {
            Size ssize = src.size(), dsize = dst.size();
            int y, x;

            for( y = range.start; y < range.end; y++ )
            {
                uchar* D = dst.data + dst.step*y;
                int sy = std::min(cvFloor(y*ify), ssize.height-1);
                const uchar* S = src.ptr(sy);

                for( x = 0; x <= dsize.width - 2; x += 2 )
                {
//                    cout<<" ori x, y: "<<x_ofs[x]<<" "<<x_ofs[x+1]<<endl;
                    uchar t0 = S[x_ofs[x]];
                    uchar t1 = S[x_ofs[x+1]];
                    D[x] = t0;
                    D[x+1] = t1;
                }

                for( ; x < dsize.width; x++ )
                {
                    D[x] = S[x_ofs[x]];
                }
            }
        }

    private:
        const Mat& src;
        Mat& dst;
        int* x_ofs;
        double ify;

        resizeNNInvoker(const resizeNNInvoker&);
        resizeNNInvoker& operator=(const resizeNNInvoker&);
    };

    static void
    resizeNN( const Mat& src, Mat& dst)
    {
        double fx = static_cast<double>(dst.cols) / src.cols;
        double fy = static_cast<double>(dst.rows) / src.rows;

        Size ssize = src.size(), dsize = dst.size();
        AutoBuffer<int> _x_ofs(dsize.width);
        int* x_ofs = _x_ofs.data();
        int pix_size = (int)src.elemSize();
        double ifx = 1./fx, ify = 1./fy;
        int x;

        for( x = 0; x < dsize.width; x++ )
        {
            int sx = cvFloor(x*ifx);
            x_ofs[x] = std::min(sx, ssize.width-1)*pix_size;
        }

        Range range(0, dsize.height);

        {
            resizeNNInvoker invoker(src, dst, x_ofs, ify);
            parallel_for_(range, invoker, dst.total()/(double)(1<<16));
        }
    }


    /**
      ******************************************************************************
      * @author         : oswin
      * @brief          : INTER_LINEAR线性插值压缩图像
      *                   计算横轴:f(R1)=u1*f11+u2*f12,f(R2)=u1*f21+u2*f22
      *                   计算纵轴:f = v1*f(R1)+v2*f(R2)
      ******************************************************************************
      */


    template<typename ST, typename DT, int bits> struct FixedPtCast
    {
        typedef ST type1;
        typedef DT rtype;
        enum { SHIFT = bits, DELTA = 1 << (bits-1) };

        DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA)>>SHIFT); }
    };

    template<typename T, typename WT, typename AT, int ONE, class VecOp>
    struct HResizeLinear
    {
        typedef T value_type;
        typedef WT buf_type;
        typedef AT alpha_type;

        void operator()(const T** src, WT** dst, int count,
                        const int* xofs, const AT* alpha,
                        int swidth, int dwidth, int cn, int xmin, int xmax ) const
        {
            int dx, k;
            VecOp vecOp;

            int dx0 = vecOp(src, dst, count,
                            xofs, alpha, swidth, dwidth, cn, xmin, xmax );

            for( k = 0; k <= count - 2; k+=2 )
            {
                const T *S0 = src[k], *S1 = src[k+1];
                WT *D0 = dst[k], *D1 = dst[k+1];
                for( dx = dx0; dx < xmax; dx++ )
                {
                    int sx = xofs[dx];
                    WT a0 = alpha[dx*2], a1 = alpha[dx*2+1];
                    WT t0 = S0[sx]*a0 + S0[sx + cn]*a1;
                    WT t1 = S1[sx]*a0 + S1[sx + cn]*a1;
                    D0[dx] = t0; D1[dx] = t1;
                }

                for( ; dx < dwidth; dx++ )
                {
                    int sx = xofs[dx];
                    D0[dx] = WT(S0[sx]*ONE); D1[dx] = WT(S1[sx]*ONE);
                }
            }

            //当f(R1)第一列rows[0](dst[0])已经在上一个循环计算过,仅需利用srows计算第二列f(R2)
            for( ; k < count; k++ )
            {
                const T *S = src[k];
                WT *D = dst[k];

                for( dx = dx0; dx < xmax; dx++ )
                {
                    int sx = xofs[dx];
                    D[dx] = S[sx]*alpha[dx*2] + S[sx+cn]*alpha[dx*2+1];
                }

                for( ; dx < dwidth; dx++ )
                    D[dx] = WT(S[xofs[dx]]*ONE);
            }
        }
    };


    template<typename T, typename WT, typename AT, class CastOp, class VecOp>
    struct VResizeLinear
    {
        typedef T value_type;
        typedef WT buf_type;
        typedef AT alpha_type;

        void operator()(const WT** src, T* dst, const AT* beta, int width ) const
        {
            WT b0 = beta[0], b1 = beta[1];
            const WT *S0 = src[0], *S1 = src[1];
            CastOp castOp;
            VecOp vecOp;

            int x = vecOp(src, dst, beta, width);
#if CV_ENABLE_UNROLLED
            for( ; x <= width - 4; x += 4 )
            {
                WT t0, t1;
                t0 = S0[x]*b0 + S1[x]*b1;
                t1 = S0[x+1]*b0 + S1[x+1]*b1;
                dst[x] = castOp(t0); dst[x+1] = castOp(t1);
                t0 = S0[x+2]*b0 + S1[x+2]*b1;
                t1 = S0[x+3]*b0 + S1[x+3]*b1;
                dst[x+2] = castOp(t0); dst[x+3] = castOp(t1);
            }
#endif
            for( ; x < width; x++ )
                dst[x] = castOp(S0[x]*b0 + S1[x]*b1);
        }
    };

    struct VResizeNoVec
    {
        template<typename WT, typename T, typename BT>
        int operator()(const WT**, T*, const BT*, int ) const
        {
            return 0;
        }
    };

    struct HResizeNoVec
    {
        template<typename T, typename WT, typename AT> inline
        int operator()(const T**, WT**, int, const int*,
                       const AT*, int, int, int, int, int) const
        {
            return 0;
        }
    };

    typedef HResizeNoVec HResizeLinearVec_8u32s;
    typedef VResizeNoVec VResizeLinearVec_32s8u;

    typedef void (*ResizeFunc)( const Mat& src, Mat& dst,
                                const int* xofs, const void* alpha,
                                const int* yofs, const void* beta,
                                int xmin, int xmax, int ksize );

    static const int MAX_ESIZE=16;
    const int INTER_RESIZE_COEF_BITS=11;
    const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;

    template <typename HResize, typename VResize>
    class resizeGeneric_Invoker :
            public ParallelLoopBody
    {
    public:
        typedef typename HResize::value_type T;
        typedef typename HResize::buf_type WT;
        typedef typename HResize::alpha_type AT;

        resizeGeneric_Invoker(const Mat& _src, Mat &_dst, const int *_xofs, const int *_yofs,
                              const AT* _alpha, const AT* __beta, const Size& _ssize, const Size &_dsize,
                              int _ksize, int _xmin, int _xmax) :
                ParallelLoopBody(), src(_src), dst(_dst), xofs(_xofs), yofs(_yofs),
                alpha(_alpha), _beta(__beta), ssize(_ssize), dsize(_dsize),
                ksize(_ksize), xmin(_xmin), xmax(_xmax)
        {
            CV_Assert(ksize <= MAX_ESIZE);
        }

        virtual void operator() (const Range& range) const CV_OVERRIDE
        {

            int dy, cn = src.channels();
            HResize hresize;
            VResize vresize;

            int bufstep = (int)alignSize(dsize.width, 16);
            AutoBuffer<WT> _buffer(bufstep*ksize);
            const T* srows[MAX_ESIZE]={nullptr};
            WT* rows[MAX_ESIZE]={nullptr};
            int prev_sy[MAX_ESIZE];

            for(int k = 0; k < ksize; k++ )
            {
                prev_sy[k] = -1;
                rows[k] = _buffer.data() + bufstep*k;//根据插值公式,取两行进行运算
            }

            const AT* beta = _beta + ksize * range.start;

            //range(0, dsize.height) dst.rows行循环
            for( dy = range.start; dy < range.end; dy++, beta += ksize )
            {
                int sy0 = yofs[dy], k0=ksize, k1=0, ksize2 = ksize/2;

                for(int k = 0; k < ksize; k++ )
                {
                    int sy = clip(sy0 - ksize2 + 1 + k, 0, ssize.height);
                    for( k1 = std::max(k1, k); k1 < ksize; k1++ )
                    {
                        if( k1 < MAX_ESIZE && sy == prev_sy[k1] ) // if the sy-th row has been computed already, reuse it.
                        {
                            if( k1 > k )
                            {
                                memcpy( rows[k], rows[k1], bufstep*sizeof(rows[0][0]) );
                                cout<<"rows: "<<" 第 "<<k1<<" 行拷贝到第 "<<k<<"行"<<" size: "<<bufstep*sizeof(rows[0][0])<<endl;
                            }
                            break;
                        }
                    }
                    if( k1 == ksize )
                        k0 = std::min(k0, k); // remember the first row that needs to be computed
                    srows[k] = src.template ptr<T>(sy);
                    prev_sy[k] = sy;
                }

                //rows[0]:f11,f21,rows[1]:f12,f22;
                //当rows[1]已经在上一个循环计算过,rows[1]拷贝给rows[0],同时进入hresize函数,k0==1;
                //仅需利用srows[1]计算一遍rows[1]:dst[k]
                if( k0 < ksize )
                    hresize( (const T**)(srows + k0), (WT**)(rows + k0), ksize - k0, xofs, (const AT*)(alpha),
                             ssize.width, dsize.width, cn, xmin, xmax );
                vresize( (const WT**)rows, (T*)(dst.data + dst.step*dy), beta, dsize.width );
            }
        }

    private:
        Mat src;
        Mat dst;
        const int* xofs, *yofs;
        const AT* alpha, *_beta;
        Size ssize, dsize;
        const int ksize, xmin, xmax;

        resizeGeneric_Invoker& operator = (const resizeGeneric_Invoker&);
    };

    template<class HResize, class VResize>
    static void resizeGeneric_( const Mat& src, Mat& dst,
                                const int* xofs, const void* _alpha,
                                const int* yofs, const void* _beta,
                                int xmin, int xmax, int ksize )
    {
        typedef typename HResize::alpha_type AT;

        const AT* beta = (const AT*)_beta;
        Size ssize = src.size(), dsize = dst.size();
        int cn = src.channels();
        ssize.width *= cn;
        dsize.width *= cn;
        xmin *= cn;
        xmax *= cn;
        // image resize is a separable operation. In case of not too strong

        Range range(0, dsize.height);
        resizeGeneric_Invoker<HResize, VResize> invoker(src, dst, xofs, yofs, (const AT*)_alpha, beta,
                                                        ssize, dsize, ksize, xmin, xmax);
        parallel_for_(range, invoker, dst.total()/(double)(1<<16));
    }


    void resize(int src_type,
                const uchar * src_data, size_t src_step, int src_width, int src_height,
                uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                double inv_scale_x, double inv_scale_y, int interpolation)
    {
        if (inv_scale_x < DBL_EPSILON || inv_scale_y < DBL_EPSILON)
        {
            inv_scale_x = static_cast<double>(dst_width) / src_width;
            inv_scale_y = static_cast<double>(dst_height) / src_height;
        }

        int  depth = CV_MAT_DEPTH(src_type), cn = CV_MAT_CN(src_type);
        Size dsize = Size(saturate_cast<int>(src_width*inv_scale_x),
                          saturate_cast<int>(src_height*inv_scale_y));
        CV_Assert( !dsize.empty() );


        static ResizeFunc linear_tab[] =
                {
                        resizeGeneric_<
                        HResizeLinear<uchar, int, short,
                        INTER_RESIZE_COEF_SCALE,
                        HResizeLinearVec_8u32s>,
                        VResizeLinear<uchar, int, short,
                        FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
                        VResizeLinearVec_32s8u>>
                };

        double scale_x = 1./inv_scale_x, scale_y = 1./inv_scale_y;

        int iscale_x = saturate_cast<int>(scale_x);
        int iscale_y = saturate_cast<int>(scale_y);

        bool is_area_fast = std::abs(scale_x - iscale_x) < DBL_EPSILON &&
                            std::abs(scale_y - iscale_y) < DBL_EPSILON;

        Mat src(Size(src_width, src_height), src_type, const_cast<uchar*>(src_data), src_step);
        Mat dst(dsize, src_type, dst_data, dst_step);

        int k, sx, sy, dx, dy;

        int xmin = 0, xmax = dsize.width, width = dsize.width*cn;
        bool area_mode = interpolation == INTER_AREA;
        bool fixpt = depth == CV_8U;
        float fx, fy;
        ResizeFunc func=0;
        int ksize=0, ksize2;
        ksize = 2, func = linear_tab[depth];
        ksize2 = ksize/2;

        CV_Assert( func != 0 );

        AutoBuffer<uchar> _buffer((width + dsize.height)*(sizeof(int) + sizeof(float)*ksize));
        int* xofs = (int*)_buffer.data();
        int* yofs = xofs + width;
        float* alpha = (float*)(yofs + dsize.height);
        short* ialpha = (short*)alpha;
        float* beta = alpha + width*ksize;
        short* ibeta = ialpha + width*ksize;
        float cbuf[MAX_ESIZE] = {0};

        for( dx = 0; dx < dsize.width; dx++ )
        {
            if( !area_mode )
            {
                fx = (float)((dx+0.5)*scale_x - 0.5);
                sx = cvFloor(fx);
                fx -= sx;
            }
            else
            {
                sx = cvFloor(dx*scale_x);
                fx = (float)((dx+1) - (sx+1)*inv_scale_x);
                fx = fx <= 0 ? 0.f : fx - cvFloor(fx);
            }

            if( sx < ksize2-1 )
            {
                xmin = dx+1;
                if( sx < 0 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
                    fx = 0, sx = 0;
            }

            if( sx + ksize2 >= src_width )
            {
                xmax = std::min( xmax, dx );
                if( sx >= src_width-1 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
                    fx = 0, sx = src_width-1;
            }

            for( k = 0, sx *= cn; k < cn; k++ )
            {
                xofs[dx*cn + k] = sx + k;
            }

            cbuf[0] = 1.f - fx;
            cbuf[1] = fx;

            if( fixpt )
            {
                for( k = 0; k < ksize; k++ )
                {
                    ialpha[dx*cn*ksize + k] = saturate_cast<short>(cbuf[k]*INTER_RESIZE_COEF_SCALE);
                }
                for( ; k < cn*ksize; k++ )
                    ialpha[dx*cn*ksize + k] = ialpha[dx*cn*ksize + k - ksize];
            }
            else
            {
                for( k = 0; k < ksize; k++ )
                    alpha[dx*cn*ksize + k] = cbuf[k];
                for( ; k < cn*ksize; k++ )
                    alpha[dx*cn*ksize + k] = alpha[dx*cn*ksize + k - ksize];
            }
        }

        for( dy = 0; dy < dsize.height; dy++ )
        {
            if( !area_mode )
            {
                fy = (float)((dy+0.5)*scale_y - 0.5);
                sy = cvFloor(fy);
                fy -= sy;
            }
            else
            {
                sy = cvFloor(dy*scale_y);
                fy = (float)((dy+1) - (sy+1)*inv_scale_y);
                fy = fy <= 0 ? 0.f : fy - cvFloor(fy);
            }

            yofs[dy] = sy;

            cbuf[0] = 1.f - fy;
            cbuf[1] = fy;

            if( fixpt )
            {
                for( k = 0; k < ksize; k++ )
                    ibeta[dy*ksize + k] = saturate_cast<short>(cbuf[k]*INTER_RESIZE_COEF_SCALE);
            }
            else
            {
                for( k = 0; k < ksize; k++ )
                    beta[dy*ksize + k] = cbuf[k];
            }
        }

        func( src, dst, xofs, fixpt ? (void*)ialpha : (void*)alpha, yofs,
              fixpt ? (void*)ibeta : (void*)beta, xmin, xmax, ksize );
    }

    void run() {
        cv::Mat src(30, 30, 0);
        cv::randu(src, 0, 25);
        src(cv::Range(3, 10), cv::Range(3, 10)).setTo(0);
        std::cout << src << std::endl;

        cv::Mat dst;
        cv::Size dsize(18, 18);

//        resizeBilinear_kernel(src, dst, dsize);
//        std::cout << dst << std::endl;
//
//        cv::Mat dst2;
//        cv::resize(src, dst2, dsize, 0, 0, cv::INTER_NEAREST);
//        std::cout << dst2 << std::endl;

//        cv::Mat dst3(dsize,CV_8U);
//        resizeNN(src,dst3);
//        std::cout<<dst3<<std::endl;

        double inv_scale_x;
        double inv_scale_y;
        int interpolation = INTER_LINEAR;
        cv::Mat dst4(dsize,CV_8U);
        resize(src.type(),src.data, src.step, src.cols, src.rows,
               dst4.data, dst4.step, dst4.cols, dst4.rows,inv_scale_x,inv_scale_y,interpolation);
        cout<<"after resizing down:\n "<<dst4<<endl;

    }

}
