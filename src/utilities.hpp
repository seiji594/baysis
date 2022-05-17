//
// utilities.hpp
// Baysis
//
// Created by Vladimir Sotskov on 22/08/2021, 22:31.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
// The object factories and registry are based on ideas in:
// [1] Alexandrescu A. Modern C++ Design: Generic Programming and Design Pattern Applied. 2001
// [2] Peleg G. Subscribing Template Classes with Object Factories in C++. 2007.
//              https://www.artima.com/articles/subscribing-template-classes-with-object-factories-in-c
//

#ifndef BAYSIS_UTILITIES_HPP
#define BAYSIS_UTILITIES_HPP

#include <map>
#include <memory>
#include <utility>
#include <tuple>
#include "../extern/typelist.hpp"


template<int I>
struct Int2Type {
    enum { Value = I };
};


// Factory subscriber for templated classes with one template arguments
template<template<class> class M, typename S, typename Factory, typename Creator>
struct FactorySubscriber_1 {
private:
    typedef FactorySubscriber_1<M, S, Factory, Creator> This_type;
    enum { EmptyList = typelist::is_tlist<S>::value && typelist::is_tlist_empty<S>::value };

    template<template<class> class M_1, typename S_1, typename Factory_1, typename Creator_1>
    static bool subscribe(Factory_1* f, Int2Type<true>) {
        return true;
    }

    template<template<class> class M_1, typename S_1, typename Factory_1, typename Creator_1>
    static bool subscribe(Factory_1* f, Int2Type<false>) {
        return f->subscribe(M_1<S_1>::Id(), Creator_1::template create<M_1<S_1> >);
    }

public:
    static bool subscribe(Factory* f) {
        return This_type::subscribe<M, S, Factory, Creator>(f, Int2Type<EmptyList>());
    }
};

// Specialisation for typelist
template<template<class> class M, typename S_H, typename... S_Ts, typename Factory, typename Creator>
struct FactorySubscriber_1<M, typelist::tlist<S_H, S_Ts...>, Factory, Creator> {
    static bool subscribe(Factory* f) {
        return FactorySubscriber_1<M, S_H, Factory, Creator>::subscribe(f) &&
                FactorySubscriber_1<M, typelist::tlist<S_Ts...>, Factory, Creator>::subscribe(f);
    }
};


// Factory subscriber for templated classes with two template arguments
template<template<class, class> class C, typename P1, typename P2, typename Factory, typename Creator>
struct FactorySubscriber_2 {
private:
    typedef FactorySubscriber_2<C, P1, P2, Factory, Creator> This_type;
    enum { EmptyList = (typelist::is_tlist<P1>::value && typelist::is_tlist_empty<P1>::value) ||
            (typelist::is_tlist<P2>::value && typelist::is_tlist_empty<P2>::value) };

    template<template<class, class> class C_1, typename P1_1, typename P2_1, typename Factory_1, typename Creator_1>
    static bool subscribe(Factory_1* f, Int2Type<true>) {
        return true;
    }

    template<template<class, class> class C_1, typename P1_1, typename P2_1, typename Factory_1, typename Creator_1>
    static bool subscribe(Factory_1* f, Int2Type<false>) {
        return f->subscribe(C_1<P1_1, P2_1>::Id(), Creator_1::template create<C_1<P1_1, P2_1> >);
    }

public:
    static bool subscribe(Factory* f) {
        return This_type::subscribe<C, P1, P2, Factory, Creator>(f, Int2Type<EmptyList>());
    }
};

// Specialisations for typelists
template<template<class, class> class C, typename P1_H, typename... P1_Ts, typename P2_H, typename... P2_Ts, typename Factory, typename Creator>
struct FactorySubscriber_2<C, typelist::tlist<P1_H, P1_Ts...>, typelist::tlist<P2_H, P2_Ts...>, Factory, Creator> {
    static bool subscribe(Factory* f) {
        return FactorySubscriber_2<C, P1_H, P2_H, Factory, Creator>::subscribe(f) &&
                FactorySubscriber_2<C, P1_H, typelist::tlist<P2_Ts...>, Factory, Creator>::subscribe(f) &&
                FactorySubscriber_2<C, typelist::tlist<P1_Ts...>, P2_H, Factory, Creator>::subscribe(f) &&
                FactorySubscriber_2<C, typelist::tlist<P1_Ts...>, typelist::tlist<P2_Ts...>, Factory, Creator>::subscribe(f);
    }
};

template<template<class, class> class C, typename P1_H, typename... P1_Ts, typename P2, typename Factory, typename Creator>
struct FactorySubscriber_2<C, typelist::tlist<P1_H, P1_Ts...>, P2, Factory, Creator> {
    static bool subscribe(Factory* f) {
        return FactorySubscriber_2<C, P1_H, P2, Factory, Creator>::subscribe(f) &&
        FactorySubscriber_2<C, typelist::tlist<P1_Ts...>, P2, Factory, Creator>::subscribe(f);
    }
};

template<template<class, class> class C, typename P1, typename P2_H, typename... P2_Ts, typename Factory, typename Creator>
struct FactorySubscriber_2<C, P1, typelist::tlist<P2_H, P2_Ts...>, Factory, Creator> {
    static bool subscribe(Factory* f) {
        return FactorySubscriber_2<C, P1, P2_H, Factory, Creator>::subscribe(f) &&
        FactorySubscriber_2<C, P1, typelist::tlist<P2_Ts...>, Factory, Creator>::subscribe(f);
    }
};


// Factory subscriber for templated classes with three template arguments (only two of them can be typelists)
template<template<class, class, class> class S, typename TM, typename OM, typename Rng,
        typename Factory, typename Creator>
struct FactorySubscriber_2a {
private:
    typedef FactorySubscriber_2a<S, TM, OM, Rng, Factory, Creator> This_type;
    enum { EmptyList = typelist::is_tlist<OM>::value && typelist::is_tlist_empty<OM>::value };

    template<template<class, class, class> class S_1, typename TM_1, typename OM_1, typename Rng_1,
            typename Factory_1, typename Creator_1>
    static bool subscribe(Factory_1* f, Int2Type<true>) {
        return true;
    }

    template<template<class, class, class> class S_1, typename TM_1, typename OM_1, typename Rng_1,
            typename Factory_1, typename Creator_1>
    static bool subscribe(Factory_1* f, Int2Type<false>) {
        return f->subscribe(S_1<TM_1, OM_1, Rng_1>::Id(), Creator_1::template create<S_1<TM_1, OM_1, Rng_1> >);
    }

public:
    static bool subscribe(Factory* f) {
        return This_type::subscribe<S, TM, OM, Rng, Factory, Creator>(f, Int2Type<EmptyList>());
    }
};

// Specialisation for typelist
template<template<class, class, class> class S, typename TM, typename OM_H, typename... OM_Ts, typename Rng,
        typename Factory, typename Creator>
struct FactorySubscriber_2a<S, TM, typelist::tlist<OM_H, OM_Ts...>, Rng, Factory, Creator> {
    static bool subscribe(Factory* f) {
        return FactorySubscriber_2a<S, TM, OM_H, Rng, Factory, Creator>::subscribe(f) &&
               FactorySubscriber_2a<S, TM, typelist::tlist<OM_Ts...>, Rng, Factory, Creator>::subscribe(f);
    }
};

template<template<class, class, class> class S, typename TM_H, typename... TM_Ts, typename OM, typename Rng,
typename Factory, typename Creator>
struct FactorySubscriber_2a<S, typelist::tlist<TM_H, TM_Ts...>, OM, Rng, Factory, Creator> {
    static bool subscribe(Factory* f) {
        return FactorySubscriber_2a<S, TM_H, OM, Rng, Factory, Creator>::subscribe(f) &&
               FactorySubscriber_2a<S, typelist::tlist<TM_Ts...>, OM, Rng, Factory, Creator>::subscribe(f);
    }
};

// Specialisation for two typelists
template<template<class, class, class> class S, typename TM_H, typename... TM_Ts, typename OM_H, typename... OM_Ts,
        typename Rng, typename Factory, typename Creator>
struct FactorySubscriber_2a<S, typelist::tlist<TM_H, TM_Ts...>, typelist::tlist<OM_H, OM_Ts...>, Rng, Factory, Creator> {
    static bool subscribe(Factory* f) {
        return FactorySubscriber_2a<S, TM_H, OM_H, Rng, Factory, Creator>::subscribe(f) &&
               FactorySubscriber_2a<S, TM_H, typelist::tlist<OM_Ts...>, Rng, Factory, Creator>::subscribe(f) &&
               FactorySubscriber_2a<S, typelist::tlist<TM_Ts...>, OM_H, Rng, Factory, Creator>::subscribe(f) &&
               FactorySubscriber_2a<S, typelist::tlist<TM_Ts...>, typelist::tlist<OM_Ts...>, Rng, Factory, Creator>::subscribe(f);
    }
};

template <typename ObjectType, typename ObjectCreator=std::function<std::shared_ptr<ObjectType>()> >
class ObjectFactory {
public:
    typedef ObjectFactory<ObjectType, ObjectCreator> This_type;
    typedef std::map<std::size_t, ObjectCreator> Creator_map;

    template<typename... Args>
    std::shared_ptr<ObjectType> create(std::size_t id, Args... args) {
        typename Creator_map::const_iterator i = this->creator_map.find(id);
        if (this->creator_map.end() != i) {
            return (i->second)(args...);
        }
        return nullptr;
    }

    bool subscribe(size_t id, ObjectCreator creator) {
        return this->creator_map.emplace(id, creator).second;
    }

    template<template<class, class, class> class S, typename TM, typename OM, typename RNG, typename Creator>
    bool subscribe() {
        return FactorySubscriber_2a<S, TM, OM, RNG, This_type, Creator>::subscribe(this);
    }

    template<template<class, class> class C, typename P1, typename P2, typename Creator>
    bool subscribe() {
        return FactorySubscriber_2<C, P1, P2, This_type, Creator>::subscribe(this);
    }

    template<template<class> class M, typename S, typename Creator>
    bool subscribe() {
        return FactorySubscriber_1<M, S, This_type, Creator>::subscribe(this);
    }

    void print() {
        Print_Map<This_type>(creator_map);
    }

private:
    Creator_map creator_map;
};


template<typename Base, typename... Args>
struct CreatorWrapper {
    template<typename Derived>
    static std::shared_ptr<Base> create(Args... args) {
        return std::make_shared<Derived>(&args...);
    }
};


//! Other compile time utilities
template<template<class> class W, typename Lst>
struct wraparound1 {
    template<std::size_t idx, typename List>
    struct map;

    template<typename H1, typename... T1s>
    struct map<0, typelist::tlist<H1, T1s...> > {
        using type = W<H1>;
        using inner_tlist = typelist::tlist<type>;
    };

    template<std::size_t k, typename H1, typename... T1s>
    struct map<k, typelist::tlist<H1, T1s...> > {
        using prev_map = map<k-1, typelist::tlist<T1s...>>;
        using type = typename prev_map::type;
        using inner_tlist = typename typelist::tlist_push_back<type, typename map<k-1, typelist::tlist<H1, T1s...> >::inner_tlist>::type;
    };

    using wrapped = typename map<Lst::size()-1, Lst>::inner_tlist;
};


template<template<class, class...> class W, typename H, typename Lst>
struct wraparound2 {
    template<std::size_t idx, typename List>
    struct map;

    template<typename H1, typename... T1s>
    struct map<0, typelist::tlist<H1, T1s...> > {
        using type = W<H, H1>;
        using inner_tlist = typelist::tlist<type>;
    };

    template<std::size_t k, typename H1, typename... T1s>
    struct map<k, typelist::tlist<H1, T1s...> > {
        using prev_map = map<k-1, typelist::tlist<T1s...>>;
        using type = typename prev_map::type;
        using inner_tlist = typename typelist::tlist_push_back<type, typename map<k-1, typelist::tlist<H1, T1s...> >::inner_tlist>::type;
    };

    using wrapped = typename map<Lst::size()-1, Lst>::inner_tlist;
};


template<template<class, class, class> class W, typename H, typename Lst, typename T>
struct wraparound3 {

    template<std::size_t idx, typename List>
    struct map;

    template<typename H1, typename... T1s>
    struct map<0, typelist::tlist<H1, T1s...> > {
        using type = W<H, H1, T>;
        using inner_tlist = typelist::tlist<type>;
    };

    template<std::size_t k, typename H1, typename... T1s>
    struct map<k, typelist::tlist<H1, T1s...> > {
        using prev_map = map<k-1, typelist::tlist<T1s...>>;
        using type = typename prev_map::type;
        using inner_tlist = typename typelist::tlist_push_back<type, typename map<k-1, typelist::tlist<H1, T1s...> >::inner_tlist>::type;
    };

    using wrapped = typename map<Lst::size()-1, Lst>::inner_tlist;
};


template<template<class> class... W>
struct zip1 {
    template<typename Lst>
    struct with {
        using list = typename typelist::tlist_flatten_list<typelist::tlist<typename wraparound1<W, Lst>::wrapped...> >::type;
    };
};

template<template<class, class...> class... W>
struct zip2 {
    template<typename H, typename Lst>
    struct with {
        using list = typename typelist::tlist_flatten_list<typelist::tlist<typename wraparound2<W, H, Lst>::wrapped...> >::type;
    };
};


template<template<class, class, class> class... W>
struct zip3 {
    template<typename H, typename Lst, typename T>
    struct with {
        using list = typename typelist::tlist_flatten_list<typelist::tlist<typename wraparound3<W, H, Lst, T>::wrapped...> >::type;
    };

    template<typename Lst1, typename Lst2, typename T>
    struct withcrossprod;

    template<typename...T1s, typename Lst2, typename T>
    struct withcrossprod<typelist::tlist<T1s...>, Lst2, T> {
        template<typename T1>
        using partial_list = typename typelist::tlist_flatten_list<typelist::tlist<typename wraparound3<W, T1, Lst2, T>::wrapped...> >::type;
        using list = typename typelist::tlist_flatten_list<typename typelist::tlist<partial_list<T1s>...>::type>::type;
    };
};


template<template<class, class...> class W, typename H, typename L1, typename L2>
struct row;

template<template<class, class...> class W, typename H, typename T1, typename...L2>
struct row<W, H, T1, typelist::tlist<L2...> > {
    using type = typelist::tlist<W<H, T1, L2>...>;
};

template<template<class, class...> class W, typename H, typename L1, typename L2, typename T>
struct rowplus;

template<template<class, class...> class W, typename H, typename T1, typename...L2, typename T>
struct rowplus<W, H, T1, typelist::tlist<L2...>, T> {
    using type = typelist::tlist<W<H, T1, L2, T>...>;
};


template<template<class, class...> class W, typename H, typename L1, typename L2>
struct crossproduct;

template<template<class, class...> class W, typename H, typename... L1s, typename... L2s>
struct crossproduct<W, H, typelist::tlist<L1s...>, typelist::tlist<L2s...>>
{
    using type = typelist::tlist<typename row<W, H, L1s, typelist::tlist<L2s...> >::type...>;
    using list = typename typelist::tlist_flatten_list<type>::type;
};

template<template<class, class...> class W, typename H, typename L1, typename L2, typename T>
struct crossproductplus;

template<template<class, class...> class W, typename H, typename... L1s, typename... L2s, typename T>
struct crossproductplus<W, H, typelist::tlist<L1s...>, typelist::tlist<L2s...>, T>
{
    using type = typelist::tlist<typename rowplus<W, H, L1s, typelist::tlist<L2s...>, T>::type...>;
    using list = typename typelist::tlist_flatten_list<type>::type;
};


// Helpers to iterate over tuples
template <class Tup, class F, std::size_t... Is>
constexpr auto static_for_impl(Tup&& t, F&& f, std::index_sequence<Is...>) {
    return (f(std::integral_constant<size_t, Is> {}, std::get<Is>(t)),...);
}

template <class... T, class F>
constexpr auto static_for(const std::tuple<T...>& t, F&& f) {
    return static_for_impl(t, std::forward<F>(f), std::make_index_sequence<sizeof...(T)>{});
}

template <class... T, class F>
constexpr auto static_for(std::tuple<T...>& t, F&& f) {
    return static_for_impl(t, std::forward<F>(f), std::make_index_sequence<sizeof...(T)>{});
}


// Integer power
static std::size_t Int_Pow(unsigned long x, long p)
{
    if (p == 0) return 1;
    if (p == 1) return x;

    unsigned long tmp = Int_Pow(x, p / 2);
    if (p%2 == 0) return tmp * tmp;
    else return x * tmp * tmp;
}

#endif //BAYSIS_UTILITIES_HPP
