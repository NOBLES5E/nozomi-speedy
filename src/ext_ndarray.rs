use ndarray::{Dimension, OwnedRepr};
use std::convert::TryInto;
use {
    crate::{Context, Readable, Reader, Writable, Writer},
    ndarray::ArrayBase,
};

impl<C, D, P> Writable<C> for ArrayBase<OwnedRepr<P>, D>
where
    C: Context,
    D: Dimension,
    P: Writable<C>,
{
    #[inline]
    fn write_to<T: ?Sized + Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
        self.shape().write_to(writer)?;
        let data_slice = self.as_slice().unwrap();
        data_slice.write_to(writer)
    }

    #[inline]
    fn bytes_needed(&self) -> Result<usize, C::Error> {
        let data_slice = self.as_slice().unwrap();
        Ok(Writable::<C>::bytes_needed(self.shape())? + Writable::<C>::bytes_needed(data_slice)?)
    }
}

impl<'a, C, P> Readable<'a, C> for ArrayBase<OwnedRepr<P>, ndarray::Dim<[usize; 2]>>
where
    C: Context,
    P: Readable<'a, C>,
{
    #[inline]
    fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        let shape: Vec<usize> = Readable::read_from(reader)?;
        let shape: [usize; 2] = shape.as_slice().try_into().unwrap();
        let data: Vec<P> = Readable::read_from(reader)?;
        Ok(unsafe { Self::from_shape_vec_unchecked(shape, data) })
    }

    #[inline]
    fn minimum_bytes_needed() -> usize {
        8
    }
}

impl<'a, C, P> Readable<'a, C> for ArrayBase<OwnedRepr<P>, ndarray::Dim<[usize; 1]>>
where
    C: Context,
    P: Readable<'a, C>,
{
    #[inline]
    fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        let shape: Vec<usize> = Readable::read_from(reader)?;
        let shape: [usize; 1] = shape.as_slice().try_into().unwrap();
        let data: Vec<P> = Readable::read_from(reader)?;
        Ok(unsafe { Self::from_shape_vec_unchecked(shape, data) })
    }

    #[inline]
    fn minimum_bytes_needed() -> usize {
        8
    }
}
