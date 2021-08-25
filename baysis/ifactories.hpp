//
// ifactories.hpp
// Baysis
//
// Created by Vladimir Sotskov on 22/08/2021, 22:31.
// Copyright © 2021 Vladimir Sotskov. All rights reserved.
// Based on ideas in:
// [1] Alexandrescu A. Modern C++ Design: Generic Programming and Design Pattern Applied. 2001
// [2] Peleg G. Subscribing Template Classes with Object Factories in C++. 2007.
//              https://www.artima.com/articles/subscribing-template-classes-with-object-factories-in-c
//

#ifndef BAYSIS_IFACTORIES_HPP
#define BAYSIS_IFACTORIES_HPP

#include <map>
#include <memory>
#include "../extern/typelist.hpp"


template<int I>
struct Int2Type {
    enum { Value = I };
};


// Factory subscriber for templated classes with one template arguments
template<std::size_t id, template<class> class M, typename S, typename Factory, typename Creator>
struct FactorySubscriber_1 {
private:
    typedef FactorySubscriber_1<id, M, S, Factory, Creator> This_type;
    enum { EmptyList = typelist::is_tlist<S>::value && typelist::is_tlist_empty<S>::value };

    template<template<class> class M_1, typename S_1, typename Factory_1, typename Creator_1>
    static bool subscribe(Factory_1* f, Int2Type<true>) {
        return true;
    }

    template<template<class> class M_1, typename S_1, typename Factory_1, typename Creator_1>
    static bool subscribe(Factory_1* f, Int2Type<false>) {
        return f->subscribe(id, Creator_1::template create<M_1<S_1> >);
    }

public:
    static bool subscribe(Factory* f) {
        return This_type::subscribe<M, S, Factory, Creator>(f, Int2Type<EmptyList>());
    }
};

// Specialisation for typelist
template<std::size_t id,
        template<class> class M, typename S_H, typename... S_Ts, typename Factory, typename Creator>
struct FactorySubscriber_1<id, M, typelist::tlist<S_H, S_Ts...>, Factory, Creator> {
    static bool subscribe(Factory* f) {
        return FactorySubscriber_1<id, M, S_H, Factory, Creator>::subscribe(f) &&
                FactorySubscriber_1<id - 1, M, typelist::tlist<S_Ts...>, Factory, Creator>::subscribe(f);
    }
};


// Factory subscriber for templated classes with three template arguments
template<std::size_t id,
        template<class, class, class> class S, typename TM, typename OM, typename Rng,
        typename Factory, typename Creator>
struct FactorySubscriber_3 {
private:
    typedef FactorySubscriber_3<id, S, TM, OM, Rng, Factory, Creator> This_type;
    enum { EmptyList = typelist::is_tlist<OM>::value && typelist::is_tlist_empty<OM>::value };

    template<template<class, class, class> class S_1, typename TM_1, typename OM_1, typename Rng_1,
            typename Factory_1, typename Creator_1>
    static bool subscribe(Factory_1* f, Int2Type<true>) {
        return true;
    }

    template<template<class, class, class> class S_1, typename TM_1, typename OM_1, typename Rng_1,
            typename Factory_1, typename Creator_1>
    static bool subscribe(Factory_1* f, Int2Type<false>) {
        return f->subscribe(id, Creator_1::template create<S_1<TM_1, OM_1, Rng_1> >);
    }

public:
    static bool subscribe(Factory* f) {
        return This_type::subscribe<S, TM, OM, Rng, Factory, Creator>(f, Int2Type<EmptyList>());
    }
};

// Specialisation for typelist
template<std::size_t id,
        template<class, class, class> class S, typename TM, typename OM_H, typename... OM_Ts, typename Rng,
        typename Factory, typename Creator>
struct FactorySubscriber_3<id, S, TM, typelist::tlist<OM_H, OM_Ts...>, Rng, Factory, Creator> {
    static bool subscribe(Factory* f) {
        return FactorySubscriber_3<id, S, TM, OM_H, Rng, Factory, Creator>::subscribe(f) &&
               FactorySubscriber_3<id - 1, S, TM, typelist::tlist<OM_Ts...>, Rng, Factory, Creator>::subscribe(f);
    }
};


template <typename ObjectType, typename ObjectCreator=std::shared_ptr<ObjectType>(*)()>
class ObjectFactory {
    typedef ObjectFactory<ObjectType, ObjectCreator> This_type;
    typedef std::map<std::size_t, ObjectCreator> Creator_map;

public:
    template<typename... Args>
    std::shared_ptr<ObjectType> create(std::size_t id, Args... args) {
        typename Creator_map::const_iterator i = this->creator_map.find(id);
        if (this->creator_map.end() != i) {
            return (i->second)(&args...);
        }
        return nullptr;
    }

    bool subscribe(size_t id, ObjectCreator creator) {
        return this->creator_map.emplace(id, creator).second;
    }

    template<template<class, class, class> class S, typename TM, typename OM, typename RNG, typename Creator>
    bool subscribe() {
        return FactorySubscriber_3<OM::size(), S, TM, OM, RNG, This_type, Creator>::subscribe(this);
    }

    template<template<class> class M, typename S, typename Creator>
    bool subscribe() {
        return FactorySubscriber_1<S::size(), M, S, This_type, Creator>::subscribe(this);
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


template<template<class, class, class> class W, typename H, typename Lst, typename T>
struct wraparound {

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


template<template<class, class, class> class... W>
struct zip {
    template<typename H, typename Lst, typename T>
    struct with {
        using list = typename typelist::tlist_flatten_list<typelist::tlist<typename wraparound<W, H, Lst, T>::wrapped...> >::type;
    };
};




#endif //BAYSIS_IFACTORIES_HPP
