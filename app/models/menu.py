"""
Database models for menus
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.orm import relationship, Mapped, mapped_column
from datetime import datetime, timezone
from app.database import Base
from typing import Optional, List


class MenuImage(Base):
    """Uploaded menu images metadata"""
    __tablename__ = "menu_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    image_hash: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True)
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ocr_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_blocks: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    total_blocks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    language_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String, default="uploaded")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc)
    )

    menu_items: Mapped[List["MenuItem"]] = relationship(
        "MenuItem",
        back_populates="menu_image",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<MenuImage(id={self.id}, filename='{self.filename}', status='{self.status}')>"

    def to_dict(self, include_ocr=False):
        """Converts to dictionary"""
        data = {
            "id": self.id,
            "filename": self.filename,
            "file_size": self.file_size,
            "image_hash": self.image_hash,
            "ocr_completed": self.ocr_completed,
            "total_blocks": self.total_blocks,
            "language": self.language,
            "language_confidence": self.language_confidence,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

        if include_ocr and self.ocr_blocks:
            data["ocr_blocks"] = self.ocr_blocks

        if self.status == "failed" and self.error_message:
            data["error_message"] = self.error_message

        return data


class MenuItem(Base):
    """Individual text blocks extracted from menu OCR"""
    __tablename__ = "menu_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    menu_image_id: Mapped[int] = mapped_column(Integer, ForeignKey("menu_images.id"), nullable=False, index=True)
    text: Mapped[str] = mapped_column(String, nullable=False, index=True)
    translated_text: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    target_language: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    original_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    converted_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    image_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    bounding_box: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(timezone.utc),
                                                 onupdate=datetime.now(timezone.utc))

    menu_image: Mapped["MenuImage"] = relationship("MenuImage", back_populates="menu_items")

    def __repr__(self):
        return f"<MenuItem(id={self.id}, text='{self.text[:30]}...')>"

    def to_dict(self):
        """Converts to dictionary"""
        return {
            "id": self.id,
            "menu_image_id": self.menu_image_id,
            "text": self.text,
            "translated_text": self.translated_text,
            "target_language": self.target_language,
            "original_price": self.original_price,
            "converted_price": self.converted_price,
            "currency": self.currency,
            "image_url": self.image_url,
            "bounding_box": self.bounding_box,
            "position": self.position,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }