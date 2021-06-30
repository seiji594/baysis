//
//  baysisException.hpp
//  Baysis
//
//  Created by Vladimir Sotskov on 11/06/2021.
//  Copyright © 2021 Vladimir Sotskov. All rights reserved.
//

/*
 * From Bayes++ the Bayesian Filtering Library
 * Copyright (c) 2003,2004,2005,2006,2011,2012,2014 Michael Stevens, Copyright (c) 2002 Michael Stevens and Australian Centre for Field Robotics
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef BAYSIS_BAYSISEXCEPTION_HPP
#define BAYSIS_BAYSISEXCEPTION_HPP

#include <exception>
#include <Eigen/Dense>


/**
 *    Base class for all exception produced by the classes
 */
class ExceptionBase : virtual public std::exception
{
public:
    const char *what() const noexcept override
    {
        return error_description;
    }
protected:
    explicit ExceptionBase (const char* description)
    {
        error_description = description;
    }
private:
    const char* error_description;
};

class LogicException : virtual public ExceptionBase
{
public:
    explicit LogicException (const char* description) :
            ExceptionBase (description)
    {}
};

class NumericException : virtual public ExceptionBase
{
public:
    explicit NumericException (const char* description) :
            ExceptionBase (description)
    {}
};


template<typename PT>
void Check_Result(const PT& res, const char* msg) {
#ifndef NDEBUG
    std::cout << msg << res << std::endl;
#endif

}

#endif //BAYSIS_BAYSISEXCEPTION_HPP
