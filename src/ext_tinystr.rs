use {
    crate::{Context, Readable, Reader, Writable, Writer},
    tinystr::{TinyStr16, TinyStr4, TinyStr8},
};

impl<'a, C> Readable<'a, C> for TinyStr4
where
    C: Context,
{
    #[inline]
    fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        Ok(unsafe { TinyStr4::new_unchecked(reader.read_u32()?) })
    }

    #[inline]
    fn minimum_bytes_needed() -> usize {
        4
    }
}

impl<C> Writable<C> for TinyStr4
where
    C: Context,
{
    #[inline]
    fn write_to<T: ?Sized + Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
        writer.write_u32(self.as_unsigned())
    }

    #[inline]
    fn bytes_needed(&self) -> Result<usize, C::Error> {
        Ok(4)
    }
}

impl<'a, C> Readable<'a, C> for TinyStr8
where
    C: Context,
{
    #[inline]
    fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        Ok(unsafe { TinyStr8::new_unchecked(reader.read_u64()?) })
    }

    #[inline]
    fn minimum_bytes_needed() -> usize {
        8
    }
}

impl<C> Writable<C> for TinyStr8
where
    C: Context,
{
    #[inline]
    fn write_to<T: ?Sized + Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
        writer.write_u64(self.as_unsigned())
    }

    #[inline]
    fn bytes_needed(&self) -> Result<usize, C::Error> {
        Ok(8)
    }
}

impl<'a, C> Readable<'a, C> for TinyStr16
where
    C: Context,
{
    #[inline]
    fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        Ok(unsafe { TinyStr16::new_unchecked(reader.read_u128()?) })
    }

    #[inline]
    fn minimum_bytes_needed() -> usize {
        16
    }
}

impl<C> Writable<C> for TinyStr16
where
    C: Context,
{
    #[inline]
    fn write_to<T: ?Sized + Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
        writer.write_u128(self.as_unsigned())
    }

    #[inline]
    fn bytes_needed(&self) -> Result<usize, C::Error> {
        Ok(16)
    }
}
