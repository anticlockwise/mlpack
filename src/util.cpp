/*
 * =====================================================================================
 *
 *       Filename:  util.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13/06/11 12:48:24
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <mlpack/util.hpp>

namespace mlpack {
    bool cmp_solution_vector(const SolutionVector &sva, const SolutionVector &svb) {
        const Event &ea = sva.event;
        const Event &eb = svb.event;
        return ea.id < eb.id;
    }
}
