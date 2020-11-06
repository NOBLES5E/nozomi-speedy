use {
    crate::{
        Context,
        Readable,
        Reader,
        Writable,
        Writer,
    },
    ndarray::ArrayBase,
    half::prelude::*,
};
use ndarray::{OwnedRepr, Dimension};
use std::convert::TryInto;

impl<C, D> Writable<C> for ArrayBase<OwnedRepr<f16>, D>
    where C: Context,
          D: Dimension,
{
    #[inline]
    fn write_to<T: ?Sized + Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
        self.shape().write_to(writer)?;
        let data_slice = self.as_slice().unwrap().reinterpret_cast();
        data_slice.write_to(writer)
    }

    #[inline]
    fn bytes_needed(&self) -> Result<usize, C::Error> {
        Ok(Writable::<C>::bytes_needed(self.shape())? +
            4 +
            self.len() * std::mem::size_of::<f16>())
    }
}

impl<'a, C> Readable<'a, C> for ArrayBase<OwnedRepr<f16>, ndarray::Dim<[usize; 2]>>
    where C: Context,
{
    #[inline]
    fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        let shape: Vec<usize> = Readable::read_from(reader)?;
        let shape: [usize; 2] = shape.try_into().unwrap();
        let data: Vec<u16> = Readable::read_from(reader)?;
        let data: Vec<f16> = data.reinterpret_into();
        Ok(unsafe { Self::from_shape_vec_unchecked(shape, data) })
    }

    #[inline]
    fn minimum_bytes_needed() -> usize {
        8
    }
}

impl<'a, C> Readable<'a, C> for ArrayBase<OwnedRepr<f16>, ndarray::Dim<[usize; 1]>>
    where C: Context,
{
    #[inline]
    fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        let shape: Vec<usize> = Readable::read_from(reader)?;
        let shape: [usize; 1] = shape.try_into().unwrap();
        let data: Vec<u16> = Readable::read_from(reader)?;
        let data: Vec<f16> = data.reinterpret_into();
        Ok(unsafe { Self::from_shape_vec_unchecked(shape, data) })
    }

    #[inline]
    fn minimum_bytes_needed() -> usize {
        8
    }
}
