use crate::private::write_length;
use std::mem;
use {
    crate::{Context, Readable, Reader, Writable, Writer},
    bytes::Bytes,
};

impl<'a, C> Readable<'a, C> for Bytes
where
    C: Context,
{
    #[inline]
    fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        let length = crate::private::read_length(reader)?;
        let result = Bytes::from(reader.read_vec(length)?);
        Ok(result)
    }

    #[inline]
    fn minimum_bytes_needed() -> usize {
        4
    }
}

impl<C> Writable<C> for Bytes
where
    C: Context,
{
    #[inline]
    fn write_to<W: ?Sized + Writer<C>>(&self, writer: &mut W) -> Result<(), C::Error> {
        write_length(self.len(), writer)?;
        writer.write_bytes(self.as_ref())
    }

    #[inline]
    fn bytes_needed(&self) -> Result<usize, C::Error> {
        unsafe_is_length!(self.len());

        let mut count = mem::size_of::<u32>();
        count += self.len();

        Ok(count)
    }
}
