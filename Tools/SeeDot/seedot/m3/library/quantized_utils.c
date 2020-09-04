// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_utils.h"

void q15_v_treesum(Q15_T* const vec, ITER_T len, SCALE_T H1, SCALE_T H2) {
  ITER_T count = len, depth = 0;
  int divbytwo = 1;

  while (depth < (H1 + H2)) {
    if (depth >= H1) {
      divbytwo = 0;
    }

    for (ITER_T p = 0; p < ((len >> 1) + 1); p++) {
      if (p < (count >> 1)) {
        if (divbytwo == 1) {
          #ifdef SHIFT
            vec[p] = (vec[2 * p] >> 1) + (vec[(2 * p) + 1] >> 1);
          #else
            vec[p] = vec[2 * p] / 2 + vec[(2 * p) + 1] / 2;
          #endif
        } else {
          vec[p] = vec[2 * p] + vec[(2 * p) + 1];
        }
      } else if ((p == (count >> 1)) && ((count & 1) == 1)) {
        if (divbytwo == 1) {
          #ifdef SHIFT
            vec[p] = (vec[2 * p] >> 1);
          #else
            vec[p] = vec[2 * p] / 2;
          #endif
        } else {
          vec[p] = vec[2 * p];
        }
      } else {
        vec[p] = 0;
      }
    }
    count = (count + 1) >> 1;
    depth++;
  }
}

void q31_v_treesum(Q31_T* const vec, ITER_T len, SCALE_T H1, SCALE_T H2) {
  ITER_T count = len, depth = 0;
  int divbytwo = 1;

  while (depth < (H1 + H2)) {
    if (depth >= H1) {
      divbytwo = 0;
    }

    for (ITER_T p = 0; p < ((len >> 1) + 1); p++) {
      if (p < (count >> 1)) {
        if (divbytwo == 1) {
          #ifdef SHIFT
            vec[p] = (vec[2 * p] >> 1) + (vec[(2 * p) + 1] >> 1);
          #else
            vec[p] = vec[2 * p] / 2 + vec[(2 * p) + 1] / 2;
          #endif
        } else {
          vec[p] = vec[2 * p] + vec[(2 * p) + 1];
        }
      } else if ((p == (count >> 1)) && ((count & 1) == 1)) {
        if (divbytwo == 1) {
          #ifdef SHIFT
            vec[p] = (vec[2 * p] >> 1);
          #else
            vec[p] = vec[2 * p] / 2;
          #endif
        } else {
          vec[p] = vec[2 * p];
        }
      } else {
        vec[p] = 0;
      }
    }
    count = (count + 1) >> 1;
    depth++;
  }
}

void q15_v_add(const Q15_T* const vec1, const Q15_T* const vec2, ITER_T len,
               Q15_T* const ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret,
               SCALE_T demote) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = (((vec1[i] >> (scvec1 + scret)) + (vec2[i] >> (scvec2 + scret))) >> demote);
    #else
      ret[i] = (((vec1[i] / scvec1) / scret) + ((vec2[i] / scvec2) / scret)) / demote;
    #endif
  }
}

void q7_v_sub(const Q7_T* const vec1, const Q7_T* const vec2, ITER_T len,
              Q7_T* const ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((vec1[i] >> (scvec1 + scret)) - (vec2[i] >> (scvec2 + scret)));
    #else
      ret[i] = ((vec1[i] / scvec1) / scret) - ((vec2[i] / scvec2) / scret);
    #endif
  }
}

void q15_v_sub(const Q15_T* const vec1, const Q15_T* const vec2, ITER_T len,
               Q15_T* const ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((vec1[i] >> (scvec1 + scret)) - (vec2[i] >> (scvec2 + scret)));
    #else
      ret[i] = ((vec1[i] / scvec1) / scret) - ((vec2[i] / scvec2) / scret);
    #endif
  }
}

void q7_v_hadamard(const Q7_T* const vec1, const Q7_T* const vec2, ITER_T len,
                   Q7_T* const ret, SCALE_T scvec1, SCALE_T scvec2) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((Q15_T)vec1[i] * (Q15_T)vec2[i]) >> (scvec1 + scvec2);
    #else
      ret[i] = ((((Q15_T)vec1[i] * (Q15_T)vec2[i]) / scvec1) / scvec2);
    #endif
  }
}

void q15_v_hadamard(const Q15_T* const vec1, const Q15_T* const vec2, ITER_T len,
                    Q15_T* const ret, SCALE_T scvec1, SCALE_T scvec2) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((Q31_T)vec1[i] * (Q31_T)vec2[i]) >> (scvec1 + scvec2);
    #else
      ret[i] = ((((Q31_T)vec1[i] * (Q31_T)vec2[i]) / scvec1) / scvec2);
    #endif
  }
}

void q15_v_sigmoid(const Q15_T* const vec, ITER_T len, Q15_T* const ret, Q15_T div,
                   Q15_T add, Q15_T sigmoid_limit, SCALE_T scale_in,
                   SCALE_T scale_out, ITER_T use_tables) {
  if (use_tables) {
    for (ITER_T i = 0; i < len; i++) {
      Q15_T x = vec[i];

      if (x <= 0) {
        Q15_T y = exp_base_16(x, 1);
        ret[i] = (Q15_T)((((Q31_T)y) << 14) / ((Q31_T)y + (Q31_T)16384));
      } else {
        ret[i] = (Q15_T)(((Q31_T)267943936L) / ((Q31_T)16384 + (Q31_T)exp_base_16(-x, 1)));
      }
    }
  } else {
    for (ITER_T i = 0; i < len; i++) {
      Q15_T x = (vec[i] / div) + add;

      if (x >= sigmoid_limit) {
        ret[i] = sigmoid_limit << (scale_out - scale_in);
      } else if (x <= 0) {
        ret[i] = 0;
      } else {
        ret[i] = x << (scale_out - scale_in);
      }
    }
  }
}

void q15_v_tanh(const Q15_T* const vec, ITER_T len, Q15_T* const ret,
                SCALE_T scale_in, SCALE_T scale_out, ITER_T use_tables) {
  if (use_tables) {
    for (ITER_T i = 0; i < len; i++) {
      Q15_T x = q15_saturate(2 * vec[i]);

      if (x <= 0) {
        Q31_T y = exp_base_16(x, 1);
        ret[i] = (Q15_T)((((Q31_T)(y - 16384)) << 14) / (y + 16384));
      } else {
        Q31_T y = exp_base_16(-x, 1);
        ret[i] = (Q15_T)((((Q31_T)(16384 - y)) << 14) / (y + 16384));
      }
    }
  } else {
    Q15_T scale = (1 << scale_in);
    for (ITER_T i = 0; i < len; i++) {
      if (vec[i] >= scale) {
        ret[i] = scale;
      } else if (vec[i] <= -scale) {
        ret[i] = (-scale);
      } else {
        ret[i] = vec[i];
      }
      ret[i] <<= (scale_out - scale_in);
    }
  }
}

void q15_v_scalar_add(Q15_T scalar, const Q15_T* const vec, ITER_T len,
                      Q15_T* const ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((scalar >> (scscalar + scret)) + (vec[i] >> (scvec + scret)));
    #else
      ret[i] = ((scalar / scscalar) / scret) + ((vec[i] / scvec) / scret);
    #endif
  }
}

void q15_v_scalar_sub(Q15_T scalar, const Q15_T* const vec, ITER_T len,
                      Q15_T* const ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((scalar >> (scscalar + scret)) - (vec[i] >> (scvec + scret)));
    #else
      ret[i] = ((scalar / scscalar) / scret) - ((vec[i] / scvec) / scret);
    #endif
  }
}

void q15_v_sub_scalar(const Q15_T* const vec, Q15_T scalar, ITER_T len,
                      Q15_T* const ret, SCALE_T scvec, SCALE_T scscalar, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((vec[i] >> (scvec + scret)) - (scalar >> (scscalar + scret)));
    #else
      ret[i] = ((vec[i] / scvec) / scret) - ((scalar / scscalar) / scret);
    #endif
  }
}

void q15_v_scalar_mul(Q15_T scalar, const Q15_T* const vec, ITER_T len,
                      Q15_T* const ret, SCALE_T scscalar, SCALE_T scvec) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((Q31_T)scalar * (Q31_T)vec[i]) >> (scscalar + scvec);
    #else
      ret[i] = ((((Q31_T)scalar * (Q31_T)vec[i]) / scscalar) / scvec);
    #endif
  }
}

void q15_v_argmax(const Q15_T* const vec, ITER_T len, ITER_T* const ret) {
  Q15_T max_value = vec[0];
  ITER_T max_index = 0;

  for (ITER_T i = 1; i < len; i++) {
    if (max_value < vec[i]) {
      max_index = i;
      max_value = vec[i];
    }
  }

  *ret = max_index;
}

void q15_v_relu(const Q15_T* const vec, ITER_T len, Q15_T* const ret) {
  for (ITER_T i = 0; i < len; i++) {
    ret[i] = q15_relu(vec[i], Q15_TMAX);
  }
}

void q15_v_exp(const Q15_T* const vec, ITER_T len, Q15_T* const ret,
               SCALE_T scvec, SCALE_T scret, ITER_T use_tables) {
  if (use_tables) {
    SCALE_T scale = scvec * scret;
    for (ITER_T i = 0; i < len; i++) {
      ret[i] = exp_base_16(vec[i], scale);
    }
  } else {
    for (ITER_T i = 0; i < len; i++) {
      ret[i] = ((Q15_T)(exp(((float)vec[i]) / scvec) * scret));
    }
  }
}

void q15_v_scale_up(const Q15_T* const vec, ITER_T len, Q15_T* const ret,
                    SCALE_T scvec) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = (vec[i] << scvec);
    #else
      ret[i] = (vec[i] * scvec);
    #endif
  }
}

void q15_v_scale_down(const Q15_T* const vec, ITER_T len, Q15_T* const ret,
                      SCALE_T scvec) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i]= (vec[i] >> scvec);
    #else
      ret[i] = (vec[i] / scvec);
    #endif
  }
}

void q15_m_transpose(const Q15_T* const mat, ITER_T nrows, ITER_T ncols,
                     Q15_T* const ret) {
  ITER_T len = nrows * ncols, counter = 0;
  for (ITER_T i = 0; i < len; i++) {
    if (counter >= len) {
      counter -= len - 1;
    }

    ret[i] = mat[counter];
    counter += nrows;
  }
}

void q15_m_reverse(const Q15_T* const mat, ITER_T nrows, ITER_T ncols, ITER_T axis,
                   Q15_T* const ret) {
  ITER_T len = nrows * ncols;

  if (axis == 0) {
    ITER_T col_counter = 0, row_index = len - ncols;

    for (ITER_T i = 0; i < len; i++) {
      if (col_counter >= ncols) {
        col_counter = 0;
        row_index -= ncols;
      }

      ret[i] = mat[row_index + col_counter];
      col_counter++;
    }
  } else {
    S_ITER_T row_counter = ncols - 1;
    ITER_T col_index = 0;

    for (ITER_T i = 0; i < len; i++) {
      if (row_counter < 0) {
        row_counter = ncols - 1;
        col_index += ncols;
      }

      ret[i] = mat[col_index + (ITER_T)row_counter];
      row_counter--;
    }
  }
}

void q15_m_add_vec(const Q15_T* const mat, const Q15_T* const vec,
                   ITER_T nrows, ITER_T ncols, Q15_T* const ret,
                   SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nrows * ncols;
  for (ITER_T i = 0, w = 0; i < len; i++, w++) {
    if (w >= ncols) {
      w = 0;
    }

    #ifdef SHIFT
      ret[i] = ((mat[i] >> (scmat + scret)) + (vec[w] >> (scvec + scret)));
    #else
      ret[i] = ((mat[i] / scmat) / scret) + ((vec[w] / scvec) / scret);
    #endif
  }
}

void q15_m_sub_vec(const Q15_T* const mat, const Q15_T* const vec,
                   ITER_T nrows, ITER_T ncols, Q15_T* const ret,
                   SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nrows * ncols;
  for (ITER_T i = 0, w = 0; i < len; i++, w++) {
    if (w >= ncols) {
      w = 0;
    }

    #ifdef SHIFT
      ret[i] = ((mat[i] >> (scmat + scret)) - (vec[w] >> (scvec + scret)));
    #else
      ret[i] = ((mat[i] / scmat) / scret) - ((vec[w] / scvec) / scret);
    #endif
  }
}

void q15_m_mulvec(const Q15_T* const mat, const Q15_T* const vec, ITER_T nrows,
                  ITER_T ncols, Q15_T* const ret, SCALE_T scmat, SCALE_T scvec,
                  SCALE_T H1, SCALE_T H2) {
  Q31_T treesumBuffer[ncols];
  for (ITER_T row = 0; row < nrows; row++) {
    Q15_T* mat_offset = (Q15_T*)mat + row * ncols;

    for (ITER_T col = 0; col < ncols; col++) {
      treesumBuffer[col] = ((Q31_T)(*mat_offset++) * (Q31_T)vec[col]);
    }

    q31_v_treesum(&treesumBuffer[0], ncols, H1, H2);
    #ifdef SHIFT
      ret[row] = (treesumBuffer[0] >> (scmat + scvec));
    #else
      ret[row] = ((treesumBuffer[0] / scmat) / scvec);
    #endif
  }
}

void q15xq7_q15_m_mulvec(const Q15_T* const mat, const Q7_T* const vec,
                         ITER_T nrows, ITER_T ncols, Q15_T* const ret,
                         SCALE_T scmat, SCALE_T scvec, SCALE_T H1, SCALE_T H2) {
  Q31_T treesumBuffer[ncols];
  for (ITER_T row = 0; row < nrows; row++) {
    Q15_T* mat_offset = (Q15_T*)mat + row * ncols;

    for (ITER_T col = 0; col < ncols; col++) {
      treesumBuffer[col] = ((Q31_T)(*mat_offset++) * (Q31_T)vec[col]);
    }

    q31_v_treesum(&treesumBuffer[0], ncols, H1, H2);
    #ifdef SHIFT
      ret[row] = (treesumBuffer[0] >> (scmat + scvec));
    #else
      ret[row] = ((treesumBuffer[0] / scmat) / scvec);
    #endif
  }
}

void q15_m_sparse_mulvec(const ITER_T* const col_indices, const Q15_T* const mat_values,
                         const Q15_T* const vec, ITER_T ndims, Q15_T* const ret,
                         SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T iter_index = 0, iter_value = 0;
  for (ITER_T k = 0; k < ndims; k++) {
    ITER_T index = col_indices[iter_index];

    while (index != 0) {
      #ifdef SHIFT
        ret[index - 1] += (((Q31_T)mat_values[iter_value] * (Q31_T)vec[k]) >> (scmat + scvec + scret));
      #else
        ret[index - 1] += (((Q31_T)mat_values[iter_value] * (Q31_T)vec[k]) / ((Q31_T)scmat * (Q31_T)scvec * (Q31_T)scret));
      #endif
      iter_index++;
      iter_value++;
      index = col_indices[iter_index];
    }

    iter_index++;
  }
}

void q7_t_add(const Q7_T* const ten1, const Q7_T* const ten2,
              ITER_T nbatches, ITER_T nrows, ITER_T ncols,
              ITER_T nchannels, Q7_T* const ret, SCALE_T scten1,
              SCALE_T scten2, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((ten1[i] >> (scten1 + scret)) + (ten2[i] >> (scten2 + scret)));
    #else
      ret[i] = ((ten1[i] / scten1) / scret) + ((ten2[i] / scten2) / scret);
    #endif
  }
}

void q15_t_add(const Q15_T* const ten1, const Q15_T* const ten2,
               ITER_T nbatches, ITER_T nrows, ITER_T ncols,
               ITER_T nchannels, Q15_T* const ret, SCALE_T scten1,
               SCALE_T scten2, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      ret[i] = ((ten1[i] >> (scten1 + scret)) + (ten2[i] >> (scten2 + scret)));
    #else
      ret[i] = ((ten1[i] / scten1) / scret) + ((ten2[i] / scten2) / scret);
    #endif
  }
}

void q15_t_add_vec(const Q15_T* const ten, const Q15_T* const vec,
                   ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                   ITER_T nchannels, Q15_T* const ret, SCALE_T scten,
                   SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  for (ITER_T i = 0, c = 0; i < len; i++, c++) {
    if (c >= nchannels) {
      c = 0;
    }

    #ifdef SHIFT
      ret[i] = ((ten[i] >> (scten + scret)) + (vec[c] >> (scvec + scret)));
    #else
      ret[i] = ((ten[i] / scten) / scret) + ((vec[c] / scvec) / scret);
    #endif
  }
}

void q7xq15_q7_t_add_vec(const Q7_T* const ten, const Q15_T* const vec,
                            ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                            ITER_T nchannels, Q7_T* const ret, SCALE_T scten,
                            SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  for (ITER_T i = 0, c = 0; i < len; i++, c++) {
    if (c >= nchannels) {
      c = 0;
    }

    #ifdef SHIFT
      ret[i] = ((ten[i] >> (scten + scret)) + (vec[c] >> (scvec + scret)));
    #else
      ret[i] = ((ten[i] / scten) / scret) + ((vec[c] / scvec) / scret);
    #endif
  }
}

void q15_t_sub_vec(const Q15_T* const ten, const Q15_T* const vec,
                   ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                   ITER_T nchannels, Q15_T* const ret, SCALE_T scten,
                   SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  for (ITER_T i = 0, c = 0; i < len; i++, c++) {
    if (c >= nchannels) {
      c = 0;
    }

    #ifdef SHIFT
      ret[i] = ((ten[i] >> (scten + scret)) - (vec[c] >> (scvec + scret)));
    #else
      ret[i] = ((ten[i] / scten) / scret) - ((vec[c] / scvec) / scret);
    #endif
  }
}

void q7_t_relu(const Q7_T* const ten, ITER_T nbatches, ITER_T nrows,
               ITER_T ncols, ITER_T nchannels, Q7_T* const ret, Q7_T limit,
               Q7_T div) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  for (ITER_T i = 0; i < len; i++) {
    ret[i] = q7_relu(ten[i], limit) / div;
  }
}

void q15_t_l2_norm(const Q15_T* const ten, ITER_T nbatches, ITER_T nrows,
                   ITER_T ncols, ITER_T nchannels, Q15_T* const ret,
                   SCALE_T scale_in, SCALE_T scale_out) {
  ITER_T len = nbatches * nrows * ncols;
  #ifndef SHIFT
    SCALE_T scdiv = (1 << scale_out);
  #endif
  for (ITER_T i = 0, j = 0; i < len; i++, j += nchannels) {
    Q31_T sum_square = 0;

    for (ITER_T c = 0; c < nchannels; c++) {
      Q31_T x = ten[j + c];
      sum_square += ((x * x) >> (2 * scale_out));
    }

    Q15_T low = 1;
    Q15_T high = (1 << (scale_out - 1));
    Q31_T one = (1 << (-(2 * scale_in + 2)));

    while (low + 1 < high) {
      Q15_T mid = ((high + low) >> 1);

      if ((int64_t)sum_square * mid * mid > one) {
        high = mid;
      } else {
        low = mid;
      }
    }

    Q15_T inverse_norm = low;

    for (ITER_T c = 0; c < nchannels; c++) {
      #ifdef SHIFT
        ret[j + c]  = (ten[j + c] >> scale_out) * inverse_norm;
      #else
        ret[j + c]  = (ten[j + c] / scdiv) * inverse_norm;
      #endif
    }
  }
}

void q15_maxpool(const Q15_T* const input, Q15_T* const output, ITER_T N,
  ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF, ITER_T WF, ITER_T CF, ITER_T COut,
  ITER_T HOut, ITER_T WOut, ITER_T G, S_ITER_T HPadU, S_ITER_T HPadD,
  S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride, ITER_T WStride,
  ITER_T HDilation, ITER_T WDilation, SCALE_T scinput, SCALE_T scoutput) {

  S_ITER_T HOffsetFL = ((HF - 1) >> 1);
  S_ITER_T HOffsetFR = (HF >> 1);
  S_ITER_T WOffsetFL = ((WF - 1) >> 1);
  S_ITER_T WOffsetFR = (WF >> 1);

  S_ITER_T HOffsetL = ((S_ITER_T)HDilation * HOffsetFL) - HPadU;
  S_ITER_T WOffsetL = ((S_ITER_T)WDilation * WOffsetFL) - WPadL;
  S_ITER_T HOffsetR = ((S_ITER_T)HDilation * HOffsetFR) - HPadD;
  S_ITER_T WOffsetR = ((S_ITER_T)WDilation * WOffsetFR) - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T hout = 0;
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; h += (S_ITER_T)HStride, hout++) {
      ITER_T wout = 0;
      ITER_T HIndexOut = hout * HOffsetOut;
      for (S_ITER_T w = WOffsetL; w < (S_ITER_T)W - WOffsetR; w += (S_ITER_T)WStride, wout++) {
        ITER_T WIndexOut = wout * WOffsetOut;
        for (ITER_T g = 0; g < G; g++) {
          ITER_T CIndexIn = g * CF;
          ITER_T CIndexOut = g * COut;
          for (ITER_T c = 0; c < COut; c++) {

            Q15_T max = Q15_TMIN;
            for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn;
              for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                ITER_T WIndexIn = ((ITER_T)woffset) * CIn;
                for (ITER_T cf = 0; cf < CF; cf++) {
                  if ((hoffset < 0) || (hoffset >= (S_ITER_T)H) || (woffset < 0) || (woffset >= (S_ITER_T)W)) {
                    if (max < 0) {
                      max = 0;
                    }
                  } else {
                    Q15_T a = input[NIndexIn + HIndexIn + WIndexIn + (cf + CIndexIn)];
                    if (max < a) {
                      max = a;
                    }
                  }
                }
              }
            }

            #ifdef SHIFT
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = (max >> (scinput + scoutput));
            #else
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = ((max / scinput) / scoutput);
            #endif
          }
        }
      }
    }
  }
}

void q15_convolution(const Q15_T* const input, const Q15_T* const filter,
  Q15_T* const output, Q31_T* const treesumBuffer, ITER_T N, ITER_T H, ITER_T W,
  ITER_T CIn, ITER_T HF, ITER_T WF, ITER_T CF, ITER_T COut, ITER_T HOut,
  ITER_T WOut, ITER_T G, S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL,
  S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, ITER_T HDilation,
  ITER_T WDilation, SCALE_T H1, SCALE_T H2, SCALE_T scinput, SCALE_T scoutput,
  SCALE_T demote) {

  S_ITER_T HOffsetFL = ((HF - 1) >> 1);
  S_ITER_T HOffsetFR = (HF >> 1);
  S_ITER_T WOffsetFL = ((WF - 1) >> 1);
  S_ITER_T WOffsetFR = (WF >> 1);

  S_ITER_T HOffsetL = ((S_ITER_T)HDilation * HOffsetFL) - HPadU;
  S_ITER_T WOffsetL = ((S_ITER_T)WDilation * WOffsetFL) - WPadL;
  S_ITER_T HOffsetR = ((S_ITER_T)HDilation * HOffsetFR) - HPadD;
  S_ITER_T WOffsetR = ((S_ITER_T)WDilation * WOffsetFR) - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T WOffsetF = CF * COut;
  ITER_T HOffsetF = WF * WOffsetF;
  ITER_T GOffsetF = HF * HOffsetF;
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T hout = 0;
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; h += (S_ITER_T)HStride, hout++) {
      ITER_T wout = 0;
      ITER_T HIndexOut = hout * HOffsetOut;
      for (S_ITER_T w = WOffsetL; w < (S_ITER_T)W - WOffsetR; w += (S_ITER_T)WStride, wout++) {
        ITER_T WIndexOut = wout * WOffsetOut;
        for (ITER_T g = 0; g < G; g++) {
          ITER_T CIndexIn = g * CF;
          ITER_T CIndexOut = g * COut;
          ITER_T GIndexF = g * GOffsetF;
          for (ITER_T c = 0; c < COut; c++) {

            ITER_T counter = 0;
            for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn;
              ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * HOffsetF;
              for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                ITER_T WIndexIn = ((ITER_T)woffset) * CIn;
                ITER_T WIndexF = ((ITER_T)(wf + WOffsetFL)) * WOffsetF;
                for (ITER_T cf = 0; cf < CF; cf++) {
                  if ((hoffset < 0) || (hoffset >= (S_ITER_T)H) || (woffset < 0) || (woffset >= (S_ITER_T)W)) {
                    treesumBuffer[counter] = 0;
                  } else {
                    treesumBuffer[counter] = ((Q31_T)input[NIndexIn + HIndexIn + WIndexIn + (cf + CIndexIn)]) *
                      ((Q31_T)filter[GIndexF + HIndexF + WIndexF + (c + cf * COut)]);
                  }
                  counter++;
                }
              }
            }

            q31_v_treesum(&treesumBuffer[0], HF * WF * CF, H1, H2);
            #ifdef SHIFT
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = (treesumBuffer[0] >> (scinput + scoutput + demote));
            #else
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = (((treesumBuffer[0] / scinput) / scoutput) / demote);
            #endif
          }
        }
      }
    }
  }
}

void q7xq15_q15_convolution(const Q7_T* const input, const Q15_T* const filter,
  Q15_T* const output, Q31_T* const treesumBuffer, ITER_T N, ITER_T H, ITER_T W,
  ITER_T CIn, ITER_T HF, ITER_T WF, ITER_T CF, ITER_T COut, ITER_T HOut,
  ITER_T WOut, ITER_T G, S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL,
  S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, ITER_T HDilation,
  ITER_T WDilation, SCALE_T H1, SCALE_T H2, SCALE_T scinput, SCALE_T scoutput,
  SCALE_T demote) {

  S_ITER_T HOffsetFL = ((HF - 1) >> 1);
  S_ITER_T HOffsetFR = (HF >> 1);
  S_ITER_T WOffsetFL = ((WF - 1) >> 1);
  S_ITER_T WOffsetFR = (WF >> 1);

  S_ITER_T HOffsetL = ((S_ITER_T)HDilation * HOffsetFL) - HPadU;
  S_ITER_T WOffsetL = ((S_ITER_T)WDilation * WOffsetFL) - WPadL;
  S_ITER_T HOffsetR = ((S_ITER_T)HDilation * HOffsetFR) - HPadD;
  S_ITER_T WOffsetR = ((S_ITER_T)WDilation * WOffsetFR) - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T WOffsetF = CF * COut;
  ITER_T HOffsetF = WF * WOffsetF;
  ITER_T GOffsetF = HF * HOffsetF;
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T hout = 0;
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; h += (S_ITER_T)HStride, hout++) {
      ITER_T wout = 0;
      ITER_T HIndexOut = hout * HOffsetOut;
      for (S_ITER_T w = WOffsetL; w < (S_ITER_T)W - WOffsetR; w += (S_ITER_T)WStride, wout++) {
        ITER_T WIndexOut = wout * WOffsetOut;
        for (ITER_T g = 0; g < G; g++) {
          ITER_T CIndexIn = g * CF;
          ITER_T CIndexOut = g * COut;
          ITER_T GIndexF = g * GOffsetF;
          for (ITER_T c = 0; c < COut; c++) {

            ITER_T counter = 0;
            for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn;
              ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * HOffsetF;
              for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                ITER_T WIndexIn = ((ITER_T)woffset) * CIn;
                ITER_T WIndexF = ((ITER_T)(wf + WOffsetFL)) * WOffsetF;
                for (ITER_T cf = 0; cf < CF; cf++) {
                  if ((hoffset < 0) || (hoffset >= (S_ITER_T)H) || (woffset < 0) || (woffset >= (S_ITER_T)W)) {
                    treesumBuffer[counter] = 0;
                  } else {
                    treesumBuffer[counter] = ((Q31_T)input[NIndexIn + HIndexIn + WIndexIn + (cf + CIndexIn)]) *
                      ((Q31_T)filter[GIndexF + HIndexF + WIndexF + (c + cf * COut)]);
                  }
                  counter++;
                }
              }
            }

            q31_v_treesum(&treesumBuffer[0], HF * WF * CF, H1, H2);
            #ifdef SHIFT
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = (treesumBuffer[0] >> (scinput + scoutput + demote));
            #else
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = (((treesumBuffer[0] / scinput) / scoutput) / demote);
            #endif
          }
        }
      }
    }
  }
}

void q7xq15_q7_convolution(const Q7_T* const input, const Q15_T* const filter,
  Q7_T* const output, Q31_T* const treesumBuffer, ITER_T N, ITER_T H, ITER_T W,
  ITER_T CIn, ITER_T HF, ITER_T WF, ITER_T CF, ITER_T COut, ITER_T HOut,
  ITER_T WOut, ITER_T G, S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL,
  S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, ITER_T HDilation,
  ITER_T WDilation, SCALE_T H1, SCALE_T H2, SCALE_T scinput, SCALE_T scoutput,
  SCALE_T demote) {

  S_ITER_T HOffsetFL = ((HF - 1) >> 1);
  S_ITER_T HOffsetFR = (HF >> 1);
  S_ITER_T WOffsetFL = ((WF - 1) >> 1);
  S_ITER_T WOffsetFR = (WF >> 1);

  S_ITER_T HOffsetL = ((S_ITER_T)HDilation * HOffsetFL) - HPadU;
  S_ITER_T WOffsetL = ((S_ITER_T)WDilation * WOffsetFL) - WPadL;
  S_ITER_T HOffsetR = ((S_ITER_T)HDilation * HOffsetFR) - HPadD;
  S_ITER_T WOffsetR = ((S_ITER_T)WDilation * WOffsetFR) - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T WOffsetF = CF * COut;
  ITER_T HOffsetF = WF * WOffsetF;
  ITER_T GOffsetF = HF * HOffsetF;
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T hout = 0;
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; h += (S_ITER_T)HStride, hout++) {
      ITER_T wout = 0;
      ITER_T HIndexOut = hout * HOffsetOut;
      for (S_ITER_T w = WOffsetL; w < (S_ITER_T)W - WOffsetR; w += (S_ITER_T)WStride, wout++) {
        ITER_T WIndexOut = wout * WOffsetOut;
        for (ITER_T g = 0; g < G; g++) {
          ITER_T CIndexIn = g * CF;
          ITER_T CIndexOut = g * COut;
          ITER_T GIndexF = g * GOffsetF;
          for (ITER_T c = 0; c < COut; c++) {

            ITER_T counter = 0;
            for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn;
              ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * HOffsetF;
              for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                ITER_T WIndexIn = ((ITER_T)woffset) * CIn;
                ITER_T WIndexF = ((ITER_T)(wf + WOffsetFL)) * WOffsetF;
                for (ITER_T cf = 0; cf < CF; cf++) {
                  if ((hoffset < 0) || (hoffset >= (S_ITER_T)H) || (woffset < 0) || (woffset >= (S_ITER_T)W)) {
                    treesumBuffer[counter] = 0;
                  } else {
                    treesumBuffer[counter] = ((Q31_T)input[NIndexIn + HIndexIn + WIndexIn + (cf + CIndexIn)]) *
                      ((Q31_T)filter[GIndexF + HIndexF + WIndexF + (c + cf * COut)]);
                  }
                  counter++;
                }
              }
            }

            q31_v_treesum(&treesumBuffer[0], HF * WF * CF, H1, H2);
            #ifdef SHIFT
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = (treesumBuffer[0] >> (scinput + scoutput + demote));
            #else
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = (((treesumBuffer[0] / scinput) / scoutput) / demote);
            #endif
          }
        }
      }
    }
  }
}