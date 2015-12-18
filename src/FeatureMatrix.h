#ifndef ARTOS_FEATUREMATRIX_H
#define ARTOS_FEATUREMATRIX_H

#include <cstddef>
#include <cstring>
#include <cassert>
#include <Eigen/Core>

namespace ARTOS
{

typedef float FeatureScalar; /**< Default scalar type used throughout ARTOS. */


/**
* @brief Container for 3-dimensional data
*
* This class stores scalar values along 3 dimensions: rows, columns and channels.  
* The last dimension is changing fastest, i.e. the memory storage order is like follows:
*
*     (0,0,0), (0,0,1), (0,0,2), ..., (0,1,0), (0,1,1), ..., (1,0,0), (1,0,1), (1,0,2), ...
*
* This class exists primarily for three reasons:
* 1. Eigen does not have efficient support for 3-dimensional arrays with a third dimension
*    of variable size.
* 2. Sometimes we need a contiguous storage of the features of all cells in a feature
*    representation of an image.
* 3. We need a convenient interface for casting cells and channels to Eigen matrices
*    for algebraic and fast vectorized operations.
*/
template<typename Scalar>
class FeatureMatrix_
{

public:

    typedef std::size_t Index; /**< Index types for the dimensions of the matrix. */
    
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Cell; /**< Feature vector type of a single cell. */
    
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ScalarMatrix; /**< A matrix of scalar values. */
    
    typedef Eigen::Map< ScalarMatrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > ChannelMap;
    typedef Eigen::Map< const ScalarMatrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > ConstChannelMap;

    /**
    * Constructs an empty feature matrix with 0 elements.
    */
    FeatureMatrix_() : m_rows(0), m_cols(0), m_channels(0), m_size(0), m_numEl(0), m_data_p(NULL), m_data(NULL, 0, 0), m_allocated(false) {};
    
    /**
    * Constructs a feature matrix with specific dimensions.
    *
    * @param[in] rows The number of rows of the matrix.
    *
    * @param[in] cols The number of columns of the matrix.
    *
    * @param[in] channels The number of channels (usually features) of each cell of the matrix.
    */
    FeatureMatrix_(Index rows, Index cols, Index channels)
    : m_rows(rows), m_cols(cols), m_channels(channels), m_size(rows * cols * channels), m_numEl(m_size),
      m_data_p((m_size > 0) ? new Scalar[m_size] : NULL),
      m_data(m_data_p, rows, cols * channels), m_allocated(m_size > 0)
    {};
    
    /**
    * Constructs a feature matrix with specific dimensions and initializes all elments
    * with a given scalar value.
    *
    * @param[in] rows The number of rows of the matrix.
    *
    * @param[in] cols The number of columns of the matrix.
    *
    * @param[in] channels The number of channels (usually features) of each cell of the matrix.
    *
    * @param[in] value The value to initialize every element with.
    */
    FeatureMatrix_(Index rows, Index cols, Index channels, Scalar value)
    : FeatureMatrix_(rows, cols, channels)
    { this->setConstant(value); };
    
    /**
    * Constructs a feature matrix with a specific number of rows and columns and
    * initializes all cells with a constant vector.
    *
    * @param[in] rows The number of rows of the matrix.
    *
    * @param[in] cols The number of columns of the matrix.
    *
    * @param[in] cell The initial value for all cells.
    */
    FeatureMatrix_(Index rows, Index cols, const Cell & cell)
    : FeatureMatrix_(rows, cols, cell.size())
    { this->setConstant(cell); };
    
    /**
    * Constructs a feature matrix wrapping an external data container.
    *
    * This version of the constructor does not allocate an own data storage and the
    * destructor will not free any memory.
    *
    * @param[in] data Pointer to external data.
    *
    * @param[in] rows The number of rows of the matrix.
    *
    * @param[in] cols The number of columns of the matrix.
    *
    * @param[in] channels The number of channels (usually features) of each cell of the matrix.
    */
    FeatureMatrix_(Scalar * data, Index rows, Index cols, Index channels)
    : m_rows(rows), m_cols(cols), m_channels(channels), m_size(rows * cols * channels), m_numEl(m_size),
      m_data_p(data), m_data(data, rows, cols * channels), m_allocated(false)
    {};
    
    /**
    * Copies another FeatureMatrix.
    *
    * @param[in] other The feature matrix to be copied.
    */
    FeatureMatrix_(const FeatureMatrix_ & other)
    : FeatureMatrix_(other.m_rows, other.m_cols, other.m_channels)
    {
        if (other.m_data_p != NULL && this->m_data_p != NULL)
            std::memcpy(reinterpret_cast<void*>(this->m_data_p), reinterpret_cast<const void*>(other.m_data_p), sizeof(Scalar) * this->m_size);
    };
    
    /**
    * Moves the data of another FeatureMatrix to this one and leaves the other
    * one empty.
    *
    * @param[in] other The feature matrix whose data is to be moved.
    */
    FeatureMatrix_(FeatureMatrix_ && other)
    : m_rows(other.m_rows), m_cols(other.m_cols), m_channels(other.m_channels), m_size(other.m_size), m_numEl(other.m_numEl),
      m_data_p(other.m_data_p), m_data(m_data_p, m_rows, m_cols * m_channels), m_allocated(other.m_allocated)
    {
        other.m_rows = other.m_cols = other.m_channels = other.m_size = 0;
        other.m_data_p = NULL;
        other.m_allocated = false;
    };
    
    /**
    * Copies another FeatureMatrix based on another scalar type by casting its elements
    * to the type of this FeatureMatrix.
    *
    * @param[in] other The feature matrix to be copied.
    */
    template<typename OtherScalar>
    FeatureMatrix_(const FeatureMatrix_<OtherScalar> & other)
    : FeatureMatrix_(other.rows(), other.cols(), other.channels())
    {
        if (!other.empty() && this->m_data_p != NULL)
        {
            Scalar * myData = this->m_data_p, * myDataEnd = myData + this->m_size;
            const OtherScalar * otherData = other.raw();
            for (; myData != myDataEnd; myData++, otherData++)
                *myData = static_cast<Scalar>(*otherData);
        }
    };
    
    virtual ~FeatureMatrix_() { if (this->m_allocated) delete[] this->m_data_p; };
    
    /**
    * Copies the contents of another feature matrix to this one.
    *
    * @param[in] other The feature matrix to be copied.
    */
    virtual FeatureMatrix_ & operator=(const FeatureMatrix_ & other)
    {
        if (this == &other)
            return *this;
        
        this->resize(other.m_rows, other.m_cols, other.m_channels);
        if (this->numEl() > 0)
            std::memcpy(reinterpret_cast<void*>(this->m_data_p), reinterpret_cast<const void*>(other.m_data_p), sizeof(Scalar) * this->numEl());
        return *this;
    };
    
    /**
    * Moves the contents of another feature matrix to this one and leaves the other one empty.
    *
    * @param[in] other The feature matrix whose data is to be moved.
    */
    virtual FeatureMatrix_ & operator=(FeatureMatrix_ && other)
    {
        if (this == &other)
            return *this;
        
        if (this->m_allocated)
            delete[] this->m_data_p;
        
        this->m_rows = other.m_rows;
        this->m_cols = other.m_cols;
        this->m_channels = other.m_channels;
        this->m_size = other.m_size;
        this->m_numEl = other.m_numEl;
        this->m_data_p = other.m_data_p;
        this->m_allocated = other.m_allocated;
        new (&(this->m_data)) Eigen::Map<ScalarMatrix>(this->m_data_p, this->m_rows, this->m_cols * this->m_channels);
        
        other.m_rows = other.m_cols = other.m_channels = other.m_size = 0;
        other.m_data_p = NULL;
        other.m_allocated = false;
        
        return *this;
    };
    
    /**
    * Copies the contents of another feature matrix based on another scalar type by casting
    * its elements to the scalar type of this FeatureMatrix.
    *
    * @param[in] other The feature matrix to be copied.
    */
    template<typename OtherScalar>
    FeatureMatrix_ & operator=(const FeatureMatrix_<OtherScalar> & other)
    {
        this->resize(other.rows(), other.cols(), other.channels());
        if (!other.empty() && this->m_data_p != NULL)
        {
            Scalar * myData = this->m_data_p, * myDataEnd = myData + this->numEl();
            const OtherScalar * otherData = other.raw();
            for (; myData != myDataEnd; myData++, otherData++)
                *myData = static_cast<Scalar>(*otherData);
        }
        return *this;
    };
    
    /**
    * @return Returns true if this feature matrix has no elements.
    */
    bool empty() const { return (this->m_data_p == NULL || this->m_numEl == 0); };
    
    /**
    * @return Returns the number of rows of this feature matrix.
    */
    Index rows() const { return this->m_rows; };
    
    /**
    * @return Returns the number of columns of this feature matrix.
    */
    Index cols() const { return this->m_cols; };
    
    /**
    * @return Returns the number of channels (usually features) of this feature matrix.
    */
    Index channels() const { return this->m_channels; };
    
    /**
    * @return Returns the number of elements in this feature matrix, i.e. rows * cols * channels.
    */
    Index numEl() const { return this->m_numEl; };
    
    /**
    * @return Returns the number of cells in this feature matrix, i.e. rows * cols.
    */
    Index numCells() const { return this->m_rows * this->m_cols; };
    
    /**
    * Changes the size of this feature matrix.
    *
    * New memory will only be allocated if rows * cols * channels is greater than before.
    * In that case, existing data will be lost.
    *
    * @param[in] rows The new number of rows.
    *
    * @param[in] cols The new number of columns.
    *
    * @param[in] channels The new number of channels.
    */
    void resize(Index rows, Index cols, Index channels)
    {
        Index numEl = rows * cols * channels;
        if (numEl > this->m_size)
        {
            if (this->m_allocated)
                delete[] this->m_data_p;
            this->m_data_p = new Scalar[numEl];
            this->m_allocated = true;
            this->m_size = numEl;
        }
        this->m_rows = rows;
        this->m_cols = cols;
        this->m_channels = channels;
        this->m_numEl = numEl;
        new (&(this->m_data)) Eigen::Map<ScalarMatrix>(this->m_data_p, this->m_rows, this->m_cols * this->m_channels);
    };
    
    /**
    * Shrink the memory allocated by this feature matrix to its current size.
    *
    * Normally, when resize() is called to resize the matrix to a smaller size, memory won't be
    * re-allocated, so that a portion of the allocated memory won't be used with the new size.
    * This method shrinks the allocated memory to the actual size of the matrix.
    */
    void shrink()
    {
        if (this->m_size > this->numEl())
        {
            Scalar * newData = new Scalar[this->numEl()];
            std::memcpy(reinterpret_cast<void*>(newData), reinterpret_cast<const void*>(this->m_data_p), sizeof(Scalar) * this->numEl());
            if (this->m_allocated)
                delete[] this->m_data_p;
            this->m_data_p = newData;
            this->m_allocated = true;
            this->m_size = this->numEl();
            new (&(this->m_data)) Eigen::Map<ScalarMatrix>(this->m_data_p, this->m_rows, this->m_cols * this->m_channels);
        }
    }
    
    /**
    * Crops this feature matrix to a sub-block.
    *
    * @param[in] firstRow The index of the row to be the first one after cropping.
    *
    * @param[in] firstCol The index of the column to be the first one after cropping.
    *
    * @param[in] numRows The number of rows after cropping.
    *
    * @param[in] numRows The number of columns after cropping.
    */
    void crop(Index firstRow, Index firstCol, Index numRows, Index numCols)
    {
        if (firstRow == 0 && firstCol == 0 && numRows == this->m_rows && numCols == this->m_cols)
            return;
        assert(firstRow >= 0 && firstCol >= 0 && numRows >= 0 && numCols >= 0
               && firstRow + numRows <= this->m_rows && firstCol + numCols <= this->m_cols);
        
        if (firstRow == 0 && firstCol == 0 && numCols == this->m_cols)
        {
            this->resize(numRows, numCols, this->m_channels);
            return;
        }
        
        Eigen::Map<ScalarMatrix> oldData(this->m_data);
        new (&(this->m_data)) Eigen::Map<ScalarMatrix>(this->m_data_p, numRows, numCols * this->m_channels);
        this->m_data = oldData.block(firstRow, firstCol * this->m_channels, numRows, numCols * this->m_channels);
        this->resize(numRows, numCols, this->m_channels);
    };
    
    /**
    * Crops this feature matrix to a sub-block.
    *
    * @param[in] firstRow The index of the row to be the first one after cropping.
    *
    * @param[in] firstCol The index of the column to be the first one after cropping.
    *
    * @param[in] firstChannel The index of the channel to be the first one after cropping.
    *
    * @param[in] numRows The number of rows after cropping.
    *
    * @param[in] numRows The number of columns after cropping.
    *
    * @param[in] numChannels The number of channels after cropping.
    */
    void crop(Index firstRow, Index firstCol, Index firstChannel, Index numRows, Index numCols, Index numChannels)
    {
        if (firstRow == 0 && firstCol == 0 && firstChannel == 0
                && numRows == this->m_rows && numCols == this->m_cols && numChannels == this->m_channels)
            return;
        assert(firstRow >= 0 && firstCol >= 0 && firstChannel >= 0 && numRows >= 0 && numCols >= 0 && numChannels >= 0
               && firstRow + numRows <= this->m_rows && firstCol + numCols <= this->m_cols && firstChannel + numChannels <= this->m_channels);
        
        Eigen::Map<ScalarMatrix> oldData(this->m_data);
        new (&(this->m_data)) Eigen::Map<ScalarMatrix>(this->m_data_p, numRows, numCols * numChannels);
        for (Index i = 0; i < numCols; i++)
            this->m_data.block(0, i * numChannels, numRows, numChannels)
                = oldData.block(firstRow, (firstCol + i) * this->m_channels + firstChannel, numRows, numChannels);
        this->resize(numRows, numCols, numChannels);
    };
    
    /**
    * Sets all elements of the feature matrix to a constant value.
    */
    void setConstant(const Scalar val) { this->m_data.setConstant(val); };
    
    /**
    * Sets all cells of the feature matrix to a constant vector.
    */
    void setConstant(const Cell & cell)
    {
        assert(cell.size() == this->m_channels);
        this->m_data = cell.transpose().replicate(this->m_rows, this->m_cols);
    };
    
    /**
    * Sets all elements of the feature matrix to 0.
    */
    void setZero() { this->setConstant(static_cast<Scalar>(0)); };
        
    /**
    * @return Returns a pointer to the raw data storage of this feature matrix.
    * The returned pointer may be NULL if the matrix is empty.
    */
    Scalar * raw() { return this->m_data_p; };
    
    /**
    * @return Returns a const pointer to the raw data storage of this feature matrix.
    * The returned pointer may be NULL if the matrix is empty.
    */
    const Scalar * raw() const { return this->m_data_p; };
    
    /**
    * Returns an Eigen::Map object wrapping the raw data of this feature matrix, where the
    * channels are stored in a contiguous way, i.e. the returned object will have as many
    * rows as this matrix, but cols * channels columns.
    */
    Eigen::Map<ScalarMatrix> data() { return this->m_data; };
    
    /**
    * Returns a constant Eigen::Map object wrapping the raw data of this feature matrix, where
    * the channels are stored in a contiguous way, i.e. the returned object will have as many
    * rows as this matrix, but cols * channels columns.
    */
    Eigen::Map<const ScalarMatrix> data() const
    { return Eigen::Map<const ScalarMatrix>(this->m_data_p, this->m_rows, this->m_cols * this->m_channels); };
    
    /**
    * Provides a linear view on this matrix as by concatenating all cells in row-major order.
    *
    * @return Returns a Eigen::Map object with rows*cols*channels rows and 1 column.
    */
    Eigen::Map<Cell> asVector() { return Eigen::Map<Cell>(this->m_data_p, this->numEl(), 1); };
    
    /**
    * Provides a linear view on this matrix as by concatenating all cells in row-major order.
    *
    * @return Returns a constant Eigen::Map object with rows*cols*channels rows and 1 column.
    */
    Eigen::Map<const Cell> asVector() const { return Eigen::Map<const Cell>(this->m_data_p, this->numEl(), 1); };
    
    /**
    * Returns an Eigen::Map object wrapping a single cell of this feature matrix.
    */
    Eigen::Map<Cell> operator()(Index i, Index j)
    {
        assert(this->m_data_p != NULL);
        assert(i >= 0 && j >= 0 && i < this->m_rows && j < this->m_cols);
        return Eigen::Map<Cell>(this->m_data_p + (i * this->m_cols + j) * this->m_channels, this->m_channels);
    };
    
    /**
    * Returns a constant Eigen::Map object wrapping a single cell of this feature matrix.
    */
    Eigen::Map<const Cell> operator()(Index i, Index j) const
    {
        assert(this->m_data_p != NULL);
        assert(i >= 0 && j >= 0 && i < this->m_rows && j < this->m_cols);
        return Eigen::Map<const Cell>(this->m_data_p + (i * this->m_cols + j) * this->m_channels, this->m_channels);
    };
    
    /**
    * Returns a reference to an element of this feature matrix.
    */
    Scalar & operator()(Index i, Index j, Index c)
    {
        assert(this->m_data_p != NULL);
        assert(i >= 0 && j >= 0 && c >= 0 && i < this->m_rows && j < this->m_cols && c < this->m_channels);
        return *(this->m_data_p + (i * this->m_cols + j) * this->m_channels + c);
    };
    
    /**
    * Returns a const reference to an element of this feature matrix.
    */
    const Scalar & operator()(Index i, Index j, Index c) const
    {
        assert(this->m_data_p != NULL);
        assert(i >= 0 && j >= 0 && c >= 0 && i < this->m_rows && j < this->m_cols && c < this->m_channels);
        return *(this->m_data_p + (i * this->m_cols + j) * this->m_channels + c);
    };
    
    /**
    * Returns an Eigen::Map object wrapping a single cell of this feature matrix,
    * addressed by a linear index.
    *
    * @param[in] c The linear index of the cell (`0 <= c < rows * cols`).
    */
    Eigen::Map<Cell> cell(Index c)
    {
        assert(this->m_data_p != NULL);
        assert(c >= 0 && c < this->m_rows * this->m_cols);
        return Eigen::Map<Cell>(this->m_data_p + c * this->m_channels, this->m_channels);
    };
    
    /**
    * Returns a constant Eigen::Map object wrapping a single cell of this feature matrix,
    * addressed by a linear index.
    *
    * @param[in] c The linear index of the cell (`0 <= c < rows * cols`).
    */
    Eigen::Map<const Cell> cell(Index c) const
    {
        assert(this->m_data_p != NULL);
        assert(c >= 0 && c < this->m_rows * this->m_cols);
        return Eigen::Map<const Cell>(this->m_data_p + c * this->m_channels, this->m_channels);
    };
    
    /**
    * Returns an Eigen::Map object wrapping a single channel (feature layer) of this matrix.
    */
    ChannelMap channel(Index c)
    {
        assert(this->m_data_p != NULL);
        assert(c >= 0 && c < this->m_channels);
        return ChannelMap(
            this->m_data_p + c, this->m_rows, this->m_cols,
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(this->m_cols * this->m_channels, this->m_channels)
        );
    };
    
    /**
    * Returns a constant Eigen::Map object wrapping a single channel (feature layer) of this matrix.
    */
    ConstChannelMap channel(Index c) const
    {
        assert(this->m_data_p != NULL);
        assert(c >= 0 && c < this->m_channels);
        return ConstChannelMap(
            this->m_data_p + c, this->m_rows, this->m_cols,
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(this->m_cols * this->m_channels, this->m_channels)
        );
    };
    
    /**
    * Adds a constant feature vector of a cell to all cells in this FeatureMatrix and
    * returns the result.
    *
    * @param[in] cell Feature vector to be added to each cell.
    *
    * @returns Returns a reference to the new matrix which forms the sum.
    */
    FeatureMatrix_ operator+(const Cell & cell) const
    {
        FeatureMatrix_ sum(*this);
        return sum += cell;
    };
    
    /**
    * Adds a constant feature vector of a cell to all cells in this FeatureMatrix.
    *
    * @param[in] cell Feature vector to be added to each cell.
    *
    * @returns Returns a reference to this FeatureMatrix.
    */
    FeatureMatrix_ & operator+=(const Cell & cell)
    {
        assert(cell.size() == this->m_channels);
        this->m_data.rowwise() += cell.colwise().replicate(this->m_cols).transpose();
        return *this;
    };
    
    /**
    * Subtracts a constant feature vector of a cell from all cells in this FeatureMatrix
    * and returns the result.
    *
    * @param[in] cell Feature vector to be subtracted from each cell.
    *
    * @returns Returns a reference to the new matrix which forms the difference.
    */
    FeatureMatrix_ operator-(const Cell & cell) const
    {
        FeatureMatrix_ sum(*this);
        return sum -= cell;
    };
    
    /**
    * Subtracts a constant feature vector of a cell from all cells in this FeatureMatrix.
    *
    * @param[in] cell Feature vector to be subtracted from each cell.
    *
    * @returns Returns a reference to this FeatureMatrix.
    */
    FeatureMatrix_ & operator-=(const Cell & cell)
    {
        assert(cell.size() == this->m_channels);
        this->m_data.rowwise() -= cell.colwise().replicate(this->m_cols).transpose();
        return *this;
    };

    /**
    * Multiplies the values of all channels in this FeatureMatrix with a scalar factor
    * depending on the channel.
    *
    * @param[in] cell Feature vector with the scalar factors to multiply each channel with.
    *
    * @returns Returns a reference to the new matrix which forms the product.
    */
    FeatureMatrix_ operator*(const Cell & cell) const
    {
        FeatureMatrix_ prod(*this);
        return prod *= cell;
    };

    /**
    * Multiplies the values of all channels in this FeatureMatrix with a scalar factor
    * depending on the channel.
    *
    * @param[in] cell Feature vector with the scalar factors to multiply each channel with.
    *
    * @returns Returns a reference to this FeatureMatrix.
    */
    FeatureMatrix_ & operator*=(const Cell & cell)
    {
        assert(cell.size() == this->m_channels);
        this->m_data.array().rowwise() *= cell.colwise().replicate(this->m_cols).transpose().array();
        return *this;
    };

    /**
    * Divides the values of all channels in this FeatureMatrix by a scalar factor
    * depending on the channel.
    *
    * @param[in] cell Feature vector with the scalar factors to divide each channel by.
    *
    * @returns Returns a reference to the new matrix which forms the quotient.
    */
    FeatureMatrix_ operator/(const Cell & cell) const
    {
        FeatureMatrix_ quot(*this);
        return quot /= cell;
    };

    /**
    * Divides the values of all channels in this FeatureMatrix by a scalar factor
    * depending on the channel.
    *
    * @param[in] cell Feature vector with the scalar factors to divide each channel by.
    *
    * @returns Returns a reference to this FeatureMatrix.
    */
    FeatureMatrix_ & operator/=(const Cell & cell)
    {
        assert(cell.size() == this->m_channels);
        this->m_data.array().rowwise() /= cell.colwise().replicate(this->m_cols).transpose().array();
        return *this;
    };


protected:
    
    Index m_rows; /**< Number of rows of this feature matrix. */
    Index m_cols; /**< Number of columns of this feature matrix. */
    Index m_channels; /**< Number of rows of this feature matrix. */
    
    Index m_size; /**< Size of the allocated array. */
    Index m_numEl; /**< Cached value of m_rows * m_cols * m_channels. */
    
    Scalar * m_data_p; /**< Pointer to the raw data. */
    
    Eigen::Map<ScalarMatrix> m_data; /**< Eigen wrapper around the data. */
    
    bool m_allocated; /**< True if this object has allocated the data storage by itself. */

};


typedef FeatureMatrix_<FeatureScalar> FeatureMatrix; /**< Feature matrix using the default scalar type. */
typedef FeatureMatrix::Cell FeatureCell; /**< Feature vector type of a single cell. */
typedef FeatureMatrix::ScalarMatrix ScalarMatrix; /**< A matrix of scalar values. */

}

#endif