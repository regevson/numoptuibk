#ifndef _OCTREESTD_HPP_
#define _OCTREESTD_HPP_

#include <array>
#include <vector>
#include <map>
#include <iostream>
#include <string>

/*
template<class Container>
struct TreeResult;

template<class T>
struct TreeResult<std::vector<T> >
{
	std::vector<T>&result;

	TreeResult(std::vector<T>&r)
	:result(r)
	{}

	void insert(const double dist_sq, const std::vector<T>&data)
	{
		result.insert( data.begin(), data.end() );
	}
};
*/

/** N dimensional space partitioning tree. Splits in the center of the space and introduces
  subspaces (childs) if a size limit of the partition container is reached.
  T the data type stored in the leaf container
  P the data type describing a space point
  N the dimension of the tree (e.g. N=3 -> octree, N=2 -> quadtree)
  @author: marcel ritter
 */
template<class T, class P, int N>
class OcTreeStd
{
	std::string m_name {""};
	size_t m_max_depth { 512 };

public:
	constexpr static int pow( int base, int exp ) noexcept
	{
		return (exp == 0 ? 1 : base * pow( base, exp -1) );
	}

struct Node
{
	P m_min_crds;
	P m_max_crds;
	P m_center;
	size_t m_max_population;
	unsigned m_depth;

	Node( const P& min_crds, const P& max_crds, size_t max_population, unsigned depth )
	: m_min_crds( min_crds)
	, m_max_crds( max_crds)
	, m_max_population( max_population )
	, m_depth( depth )
	{
		for( unsigned i = 0 ; i < N; ++i )
			m_center[i] = (max_crds[i] + min_crds[i])/2;
	}


	virtual ~Node(){}

    virtual bool isLeaf() const 
    {
        return false;
    }
	virtual bool insert( const P& crds, const T& data )  = 0;
//	virtual std::string toString( unsigned v ) const noexcept = 0;

//	template<class Container>
	virtual void getManhattenRangeCellWise( const P&min, const P&max, const double range, std::vector<T>&res ) const  = 0;
	virtual void getManhattenRangeFine( const P&min, const P&max, const double range, std::vector<T>&res ) const  = 0;
//	virtual void getManhattenRangeCellWise( const P&min, const P&max, const double range, std::multimap<double, T>&res ) const noexcept = 0;
	virtual void getManhattenRangeFine( const P&min, const P&max, const double range, std::multimap<double,T>&res ) const  = 0;
    virtual void getNodePositionsFromLevel(unsigned level, std::vector<P> &container) noexcept = 0;

	virtual void getEuclideanRangeFine( const P&center, const double range, std::multimap<double,T>&res ) const = 0;
	virtual void getEuclideanRangeFine( const P&center, const double range, std::vector<T>&res ) const = 0;

	virtual unsigned getDepth() const noexcept = 0;
	virtual std::size_t getSize() const noexcept = 0;
};

struct Branch : public Node
{
	std::array<Node*, pow(2, N) > m_childs;
	OcTreeStd<T,P,N>& m_parent;

	Branch( const P& min_crds, const P& max_crds, size_t max_population, unsigned depth, OcTreeStd<T,P,N>& parent  )
	: Node( min_crds, max_crds, max_population, depth )
	, m_parent( parent )
	{
		for( unsigned i = 0; i < pow(2, N); ++i )
			m_childs[i] = nullptr;
	}

	virtual ~Branch()
	{
		for( unsigned i = 0; i < pow(2, N); ++i )
		{
			if( m_childs[i] )
				delete m_childs[i];
		}
	}

	unsigned getDepth() const noexcept override
	{
	unsigned deapest { 0 };
		for( unsigned i = 0; i < pow(2, N) ; ++i )
			if( m_childs[i] )
				deapest = std::max<unsigned>( m_childs[i]->getDepth(), deapest );

		return deapest;
	}

	std::size_t getSize() const noexcept
	{
	std::size_t sum { 0 };
		for( unsigned i = 0; i < pow(2, N) ; ++i )
			if( m_childs[i] )
				sum += m_childs[i]->getSize();
		return sum;
	}
    
    
/*
	std::string toString( unsigned v ) const noexcept override
	{
	std:: string txt;

	unsigned count_childs = 0;
		for( unsigned i = 0; i < pow(2, N) ; ++i )
			if( m_childs[i] )
				++count_childs;

		for( unsigned i = 0; i < this->m_depth ; ++i )
			txt += "   ";

		txt += "Branch with childs: " + std::to_string( count_childs );

		if( v > 0 )
		{
			txt += ", {";
			for( unsigned i = 0; i < N-1; ++i )
			{
				txt += std::to_string(this->m_min_crds[i]);
				if( i < N )
					txt += ", ";
			}
			txt += "}-{";
			for( unsigned i = 0; i < N-1; ++i )
			{
				txt += std::to_string(this->m_max_crds[i]);
				if( i < N )
					txt += ", ";
			}
			txt += "}\n";
		}
		else
			txt += "\n";

		for( unsigned i = 0; i < pow(2, N) ; ++i )
			if( m_childs[i] )
				txt += m_childs[i]->toString( v );

		return txt;
	}
*/

	bool insert( const P& crds, const T& data ) noexcept override
	{
		if( this->m_depth >= m_parent.m_max_depth )
		{
			printf( "WARNING: Max depth %llu reached in octree %s\n", this->m_depth, m_parent.m_name.c_str() );
			std::cout << /* "    for data: " << data << */ "    at: ";
			for( unsigned i = 0; i < N; ++i ) 
				std::cout << crds[i] << ", ";
			std::cout << std::endl;
			
			return false;
		}
		const P& center = this->m_center;
		unsigned index = 0;
		P min;
		P max;

		for( unsigned i = 0 ; i < N; ++i )
		{
			//std::cout << crds[i] << ", " << center[i] << std::endl;

			if( crds[i] > center[i] )
			{
				index += 1 << i;
				min[i] = center[i];
				max[i] = this->m_max_crds[i];
			}
			else
			{
				min[i] = this->m_min_crds[i];
				max[i] = center[i];
			}
		}

		//std::cout << "Branch computes cell index: " << index << std::endl;

		if( m_childs[index] )
		{
			//std::cout << "fill existing leaf" << std::endl;
			// insert and possibly split
			if( !m_childs[index]->insert( crds, data ) )
			{
				//std::cout << "child split & copy" << std::endl;
				Branch* tmpb = new Branch( m_childs[index]->m_min_crds, m_childs[index]->m_max_crds, m_childs[index]->m_max_population,this->m_depth+1, this->m_parent ) ;
				Leaf* tmpl = static_cast<Leaf*> ( m_childs[index] );

				for( unsigned i = 0; i < tmpl->m_data.size() ; ++i )
				{
					//std::cout << i  << "/" << tmpl->m_data.size() << std::endl;
					tmpb->insert( tmpl->m_pos[i], tmpl->m_data[i] );
				}

				tmpb->insert( crds, data );

				delete tmpl;
				m_childs[index] = tmpb;

				return true;
			}
			//else
			//{
			//	//std::cout << "filled" << std::endl;
			//}
		}
		else
		{
			//std::cout << "create new leaf" << std::endl;
			m_childs[index] = new Leaf( min, max, this->m_max_population, this->m_depth+1 );
			m_childs[index]->insert( crds, data );
		}

		return true;
	}
//	template<class Container>
	void getManhattenRangeCellWise( const P&min, const P&max, const double range, std::vector<T>&res ) const noexcept
	{
	bool touching { true };

		for( unsigned i = 0; i < N; ++i )
			if( this->m_max_crds[i] < min[i] || this->m_min_crds[i] >= max[i] )
			{
				touching = false;
				break;
			}

		if( !touching )
			return;

		for( unsigned i = 0; i < pow(2, N) ; ++i )
			if( m_childs[i] )
				m_childs[i]->getManhattenRangeCellWise( min, max, range, res);
	}
    
    void getNodePositionsFromLevel(unsigned level, std::vector<P> &container) noexcept override
    {
        if (this->m_depth == level) {
            //we're right -> use these
            container.push_back(this->m_center);
        }
        else if(level > this->m_depth) {
            for(unsigned i=0; i < pow(2,N); ++i) {
                if(m_childs[i]) {
                    m_childs[i]->getNodePositionsFromLevel(level, container);
                }
            }
        }
    }

	void getManhattenRangeFine( const P&min, const P&max, const double range, std::vector<T>&res ) const noexcept
	{
	bool touching { true };

		for( unsigned i = 0; i < N; ++i )
			if( this->m_max_crds[i] < min[i] || this->m_min_crds[i] >= max[i] )
			{
				touching = false;
				break;
			}

		if( !touching )
			return;

		for( unsigned i = 0; i < pow(2, N) ; ++i )
			if( m_childs[i] )
				m_childs[i]->getManhattenRangeFine( min, max, range, res);
	}

	void getManhattenRangeFine( const P&min, const P&max, const double range, std::multimap<double,T>&res ) const noexcept
	{
		bool touching { true };

		for( unsigned i = 0; i < N; ++i )
			if( this->m_max_crds[i] < min[i] || this->m_min_crds[i] >= max[i] )
			{
				touching = false;
				break;
			}

		if( !touching )
			return;

		for( unsigned i = 0; i < pow(2, N) ; ++i )
			if( m_childs[i] )
				m_childs[i]->getManhattenRangeFine( min, max, range, res);
	}

	void getEuclideanRangeFine( const P&center, const double range, std::multimap<double,T>&res ) const noexcept
	{
		bool touching { true };

		for( unsigned i = 0; i < N; ++i )
			if( this->m_max_crds[i] < center[i] - range || this->m_min_crds[i] >= center[i] + range )
			{
				touching = false;
				break;
			}

		if( !touching )
			return;

		for( unsigned i = 0; i < pow(2, N) ; ++i )
			if( m_childs[i] )
				m_childs[i]->getEuclideanRangeFine( center, range, res);
	}

	void getEuclideanRangeFine( const P&center, const double range, std::vector<T>&res ) const noexcept
	{
		bool touching { true };

		for( unsigned i = 0; i < N; ++i )
			if( this->m_max_crds[i] < center[i] - range || this->m_min_crds[i] >= center[i] + range )
			{
				touching = false;
				break;
			}

		if( !touching )
			return;

		for( unsigned i = 0; i < pow(2, N) ; ++i )
			if( m_childs[i] )
				m_childs[i]->getEuclideanRangeFine( center, range, res);
	}
};

struct Leaf : public Node
{
	std::vector<T> m_data;
	std::vector<P> m_pos;

	Leaf( const P& min_crds, const P& max_crds, size_t max_population, unsigned depth  )
	: Node ( min_crds, max_crds, max_population, depth)
	{
		m_data.reserve( max_population );
		m_pos.reserve( max_population );
	}

	virtual ~Leaf(){}
    
    virtual bool isLeaf() const noexcept override
    {
        return true;
    }

	unsigned getDepth() const noexcept override
	{
		return this->m_depth;
	}

	std::size_t getSize() const noexcept
	{
		return this->m_data.size();
	}

	bool insert( const P& crds, const T& data ) noexcept override
	{
		if( m_data.size() < m_data.capacity() )
		{
			m_data.push_back( data );
			m_pos.push_back( crds );
			return true;
		}

		//std::cout << "insert already full " << m_data.size() << ", " << m_data.capacity() << std::endl;
		return false;
	}
    
    void getNodePositionsFromLevel(unsigned level, std::vector<P> &container) noexcept override
    {
        //always add all children here
        for(auto &pos : m_pos) {
            container.push_back(pos);
        }
    }

/*
	std::string toString( unsigned v ) const noexcept override
	{
	std::string txt;
		for( unsigned i = 0; i < this->m_depth ; ++i ) 	txt += "   ";

		txt+= "Leave with elements: " + std::to_string( this->m_data.size() );
		if( v > 0 )
		{
			txt += ", {";
			for( unsigned i = 0; i < N-1; ++i )
			{
				txt += std::to_string(this->m_min_crds[i]);
				if( i < N )
					txt += ", ";
			}
			txt += "}-{";
			for( unsigned i = 0; i < N-1; ++i )
			{
				txt += std::to_string(this->m_max_crds[i]);
				if( i < N )
					txt += ", ";
			}
			txt += "}\n";
		}
		else
			txt += "\n";

		if( v > 1 )
		{
			for( auto elem : this->m_data )
			{
				for( unsigned i = 0; i < this->m_depth ; ++i )
					txt += "   ";
				txt += std::to_string( elem ) + "\n";
			}
		}

		return txt;
	}
*/

//	template<class Container>
	void getManhattenRangeCellWise( const P&min, const P&max, const double range, std::vector<T>&res ) const noexcept
	{
	bool touching { true };

		for( unsigned i = 0; i < N; ++i )
			if( this->m_max_crds[i] < min[i] || this->m_min_crds[i] >= max[i] )
			{
				touching = false;
				break;
			}

		if( !touching )
			return;

		res.insert( res.end(), this->m_data.begin(), this->m_data.end() );
	}


	void getManhattenRangeFine( const P&min, const P&max, const double range, std::vector<T>&res ) const noexcept
	{
	bool touching { true };

		for( unsigned i = 0; i < N; ++i )
			if( this->m_max_crds[i] < min[i] || this->m_min_crds[i] >= max[i] )
			{
				touching = false;
				break;
			}

		if( !touching )
			return;

		//std::cout << "getManhattenRangeFine check" << std::endl;
		for(unsigned j = 0; j < this->m_data.size(); ++j )
		{
		bool inside = true;
			for( unsigned i = 0; i < N; ++i )
				if( this->m_pos[j][i] < min[i] || this->m_pos[j][i] >= max[i] )
					inside = false;

			if( inside )
				res.push_back( this->m_data[j] );
		}
	}

	// avoid cpoy of the previous!
	void getManhattenRangeFine( const P&min, const P&max, const double range, std::multimap<double,T>&res ) const noexcept
	{
		bool touching { true };

		for( unsigned i = 0; i < N; ++i )
			if( this->m_max_crds[i] < min[i] || this->m_min_crds[i] >= max[i] )
			{
				touching = false;
				break;
			}

		if( !touching )
			return;

		//std::cout << "getManhattenRangeFine check" << std::endl;
		for(unsigned j = 0; j < this->m_data.size(); ++j )
		{
		bool inside = true;
			for( unsigned i = 0; i < N; ++i )
				if( this->m_pos[j][i] < min[i] || this->m_pos[j][i] >= max[i] )
					inside = false;

			if( inside )
			{
			P center;
			double distance_sq { 0.0 };

				for( unsigned i = 0; i < N; ++i  )
				{
					center[i] = ( max[i] + min[i] )/2;
					distance_sq += (this->m_pos[j][i] - center[i])*(this->m_pos[j][i] - center[i]);
				}


				res.insert( std::pair<double, T>(distance_sq, this->m_data[j]) );
			}
		}
	}

	// avoid copy of the previous!
	void getEuclideanRangeFine( const P & center,const double range, std::multimap<double,T>&res ) const noexcept
	{
		bool touching { true };

		for( unsigned i = 0; i < N; ++i )
			if( this->m_max_crds[i] < center[i] - range || this->m_min_crds[i] >= center[i] + range )
			{
				touching = false;
				break;
			}

		if( !touching )
			return;

		for(unsigned j = 0; j < this->m_data.size(); ++j )
		{
		double distance_sq {0.0};

			for( unsigned i = 0; i < N; ++i )
				distance_sq += (center[i] - this->m_pos[j][i]) * (center[i] - this->m_pos[j][i]);

			if( distance_sq <= range*range )
				res.insert( std::pair<double, T>(distance_sq, this->m_data[j]) );
		}
	}

	void getEuclideanRangeFine( const P & center,const double range, std::vector<T>&res ) const noexcept
	{
		bool touching { true };

		for( unsigned i = 0; i < N; ++i )
			if( this->m_max_crds[i] < center[i] - range || this->m_min_crds[i] >= center[i] + range )
			{
				touching = false;
				break;
			}

		if( !touching )
			return;

		for(unsigned j = 0; j < this->m_data.size(); ++j )
		{
		double distance_sq {0.0};

			for( unsigned i = 0; i < N; ++i )
				distance_sq += (center[i] - this->m_pos[j][i]) * (center[i] - this->m_pos[j][i]);

			if( distance_sq <= range*range )
				res.push_back(this->m_data[j]);
		}
	}

};

protected:
	P m_min_crds;
	P m_max_crds;
//	unsigned m_max_depth;
	unsigned m_max_population;
	Node* m_root;

public:
    template<typename traverse_trait>
    struct OctreeTraverser
    {
        traverse_trait &m_visitor;
        explicit OctreeTraverser(traverse_trait &visitor)
            : m_visitor(visitor)
        {
        }
        
        void visit(Node *n)
        {
            if(n->isLeaf()) {
                auto *l = static_cast<Leaf*>(n);
                m_visitor.visit(l);
            }
            else {
                auto *b = static_cast<Branch*>(n);
                if(m_visitor.visit(b)) {
                    for(auto *c : b->m_childs) {
                        if(c) {
                            visit(c);
                        }
                    }
                }
                
            }
        }
    };

/** Create tree by defing the space bounds (min, max), the maximum depth, and the maximum data samples
  allowed to be stored in a leave.
 */
	OcTreeStd( const P& min_crds, const P& max_crds, unsigned max_population, std::string name, size_t max_depth = 512 )
	: m_min_crds( min_crds )
	, m_max_crds( max_crds )
	, m_max_population( max_population )
	, m_name( name )
	, m_max_depth( max_depth )
    , m_root(nullptr)
	{
        clear();
	}

	virtual ~OcTreeStd()
	{
		delete m_root;
	}

	std::size_t getSize() const noexcept
	{
		if( m_root )
			return m_root->getSize();

		return 0;
	}

	unsigned getDepth() const noexcept
	{
		if( m_root )
			return m_root->getDepth();

		return 0;
	}

	unsigned getMaxPopulation() const noexcept
	{
		return m_max_population;
	}

	bool insert( const P& crds, const T& data ) noexcept
	{
		// check if point is in bounds
		for( unsigned i = 0; i < N; ++i )
			if( crds[i] < this->m_min_crds[i] || crds[i] > this->m_max_crds[i] )
			{
				//std::cout << "OcTreeStd::insert() ERROR out of bounds!" << std::endl;
				return false;
			}

		if( m_root->insert(crds, data) )
			return true;
		else // split & copy
		{
		//std::cout << "split & copy" << std::endl;
			Branch* tmpb = new Branch( this->m_min_crds, this->m_max_crds, this->m_max_population, 1, *this ) ;
			Leaf* tmpl = static_cast<Leaf*> ( m_root );

			for( unsigned i = 0; i < tmpl->m_data.size() ; ++i )
			{
				//std::cout << i  << "/" << tmpl->m_data.size() << std::endl;
				tmpb->insert( tmpl->m_pos[i], tmpl->m_data[i] );
			}

			tmpb->insert( crds, data );

			delete tmpl;
			m_root = tmpb;
		}

		return true;
	}

	bool saveInsert( const P& crds, const T& data ) noexcept
	{
		for( unsigned i = 0; i < N; ++i )
		{
			if( crds[i] != crds[i] )
			{
				//std::cout << "OcTreeStd skipped contained Nan point" << std::endl;
				return false;
			}
		}

		if( m_root )
		{
			std::multimap<double,T> res;
			getManhattenRangeFine( crds, 1.0e-8, res );
			if( res.size() > 0 )
			{
				//std::cout << "OcTreeStd skipped contained point" << std::endl;
				//std::cin.ignore();
				return false;
			}
			else
				return insert(crds, data );
		}

		return insert(crds, data );
	}
/*
	std::string toString( unsigned verbose = 0) const noexcept
	{
		return m_root->toString( verbose );
	}
*/
    void clear(const P &min, const P &max)
    {
        m_min_crds = min;
        m_max_crds = max;
        clear();
    }
    
    void clear()
    {
        //delete the root, and recreate a leaf root based on the min and max coords
        if(m_root) {
            delete m_root;
        }
		m_root = new Leaf( m_min_crds, m_max_crds, m_max_population, 1 );
    }

// tree query functions:
//	template<class Container>
	void getManhattenRangeCellWise( P min, P max, const double range, std::vector<T>&res ) const noexcept
	{
		if( m_root )
		{
			res.clear();
			m_root->getManhattenRangeCellWise( min, max, range, res );
		}
	}

	void getManhattenRangeCellWise( P pos, const double range, std::vector<T>&res ) const noexcept
	{
		P min, max;
		for(unsigned i = 0; i < N; ++i)
		{
			min[i] = pos[i] - range;
			max[i] = pos[i] + range;
		}

		getManhattenRangeCellWise( min, max, range, res);
	}


	void getManhattenRangeFine( P min, P max, const double range, std::vector<T>&res ) const noexcept
	{
		if( m_root )
		{
			res.clear();
			m_root->getManhattenRangeFine( min, max, range, res );
		}
	}

	void getManhattenRangeFine( P pos, const double range, std::vector<T>&res ) const noexcept
	{
		P min, max;
		for(unsigned i = 0; i < N; ++i)
		{
			min[i] = pos[i] - range;
			max[i] = pos[i] + range;
		}

		getManhattenRangeFine( min, max, range, res);
	}

	void getManhattenRangeFine( const P&min, const P&max, const double range, std::multimap<double,T>&res ) const noexcept
	{
		if( m_root )
		{
			res.clear();
			m_root->getManhattenRangeFine( min, max, range, res );
		}
	}


    std::vector<P> getNodePositionsFromLevel(unsigned level)
    {
    auto container = std::vector<P>();
        if (m_root) {
            m_root->getNodePositionsFromLevel(level, container);
        }
        return container;
    }
    
    template<typename traverse_traits>
    void traverse(traverse_traits &visitor)
    {
        OctreeTraverser<traverse_traits> traverser(visitor);
        if(m_root)
            traverser.visit(m_root);
    }


	void getManhattenRangeFine( P pos, const double radius, std::multimap<double,T>&res ) const noexcept
	{
	P min, max;
		for(unsigned i = 0; i < N; ++i)
		{
			min[i] = pos[i] - radius;
			max[i] = pos[i] + radius;
		}

		getManhattenRangeFine( min, max, radius, res);
	}

	void getEuclideanRangeFine( P center, const double radius, std::multimap<double,T>&res ) const noexcept
	{
		m_root->getEuclideanRangeFine( center, radius, res);
	}

	void getEuclideanRangeFine( P center, const double radius, std::vector<T>&res ) const noexcept
	{
		size_t capacity = res.capacity();
		res.clear();
		res.reserve(capacity);

		m_root->getEuclideanRangeFine( center, radius, res);
	}
};


#endif
